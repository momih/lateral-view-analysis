import argparse
from glob import glob
import os
from os.path import join, exists
import pickle

import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, split_dataset
from hemis import Hemis, add_dropout_hemis

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)


def train(data_dir, csv_path, splits_path, output_dir, target='pa', nb_epoch=100, learning_rate=1e-4, batch_size=1,
          dropout=True, pretrained=False, df=None):
    assert target in ['pa', 'l', 'joint']

    print("Training mode: {}".format(target))

    if not exists(output_dir):
        os.makedirs(output_dir)

    if not exists(splits_path):
        split_dataset(csv_path, splits_path)

    # Find device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device that will be used is: {0}'.format(device))

    # Load data
    preprocessing = Compose([Normalize(), ToTensor()])
    trainset = PCXRayDataset(data_dir, csv_path, splits_path, transform=preprocessing, pretrained=pretrained)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = PCXRayDataset(data_dir, csv_path, splits_path, transform=preprocessing, dataset='val',
                           pretrained=pretrained)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("{0} patients in training set.".format(len(trainset)))
    print("{0} patients in validation set.".format(len(valset)))

    # Load model
    if pretrained:
        in_channels = 3
    else:
        in_channels = 2 if target == 'joint' else 1
    model = Hemis(num_classes=trainset.nb_labels, in_channels=1)

    # Add dropout
    if dropout:
        model = add_dropout_hemis(model, p=0.2)

    # criterion = nn.BCELoss()
    print(trainset.labels_weights)
    criterion = nn.MultiLabelSoftMarginLoss(weight=torch.from_numpy(trainset.labels_weights).to(device))

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Used to decay learning rate

    # Resume training if possible
    start_epoch = 0
    start_batch = 0
    train_loss = []
    val_loss = []
    val_accuracy = []
    weights_files = glob(join(output_dir, '{}-e*.pt'.format(target)))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array([int(w[len(join(output_dir, '{}-e'.format(target))):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max() + 1
        weights_files = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()]

        # Find most recent batch
        if len(weights_files) > 1:
            batches = np.array([int(w[len(join(output_dir, '{}-e'.format(target))):-len('.pt')].split('i')[1]) for w in weights_files])
            start_batch = batches.max()
            weights_file = weights_files[np.argmax(batches)]
            start_epoch -= 1
        else:
            weights_file = weights_files[0]
        model.load_state_dict(torch.load(weights_file))

        with open(join(output_dir, '{}-train_loss.pkl'.format(target)), 'rb') as f:
            train_loss = pickle.load(f)

        with open(join(output_dir, '{}-val_loss.pkl'.format(target)), 'rb') as f:
            val_loss = pickle.load(f)

        with open(join(output_dir, '{}-val_accuracy.pkl'.format(target)), 'rb') as f:
            val_accuracy = pickle.load(f)

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    model.to(device)

    # Training loop
    for epoch in range(start_epoch, nb_epoch):  # loop over the dataset multiple times
        scheduler.step()

        model.train()

        running_loss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(device)
        for i, data in enumerate(trainloader, 0):
            # Skip to current batch
            if epoch == start_epoch and i < start_batch:
                continue

            if target == 'pa':
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)
            elif target == 'l':
                input, label = data['L'].to(device), data['encoded_labels'].to(device)
            else:
                pa, l, label = data['PA'].to(device), data['L'].to(device), data['encoded_labels'].to(device)
                input = torch.zeros((pa.size()[0], in_channels, pa.size()[2], pa.size()[3]), requires_grad=False,
                                    dtype=pa.dtype).to(device)
                input[:, 0] = pa[:, 0]
                input[:, 1] = l[:, 0]

            # Forward
            output = model(input)
            optimizer.zero_grad()
            loss = criterion(output, label)

            # Backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data
            print_every = max(1, len(trainset) // (20 * batch_size))
            if (i + 1) % print_every == 0:
                running_loss = running_loss.cpu().detach().numpy().squeeze() / print_every
                print('[{0}, {1:5}] loss: {2:.5f}'.format(epoch + 1, i + 1, running_loss))
                train_loss.append(running_loss)

                with open(join(output_dir, '{}-train_loss.pkl'.format(target)), 'wb') as f:
                    pickle.dump(train_loss, f)
                torch.save(model.state_dict(), join(output_dir, '{0}-e{1}-i{2}.pt'.format(target, epoch, i + 1)))
                running_loss = torch.zeros(1, requires_grad=False).to(device)

        model.eval()

        class_correct = torch.zeros(trainset.nb_labels, requires_grad=False, dtype=torch.float).to(device)
        class_total = torch.zeros(trainset.nb_labels, requires_grad=False, dtype=torch.float).to(device)
        running_loss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(device)
        val_preds = []
        val_true = []
        for i, data in enumerate(valloader, 0):
            if target == 'pa':
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)
            elif target == 'l':
                input, label = data['L'].to(device), data['encoded_labels'].to(device)
            else:
                pa, l, label = data['PA'].to(device), data['L'].to(device), data['encoded_labels'].to(device)
                input = torch.zeros((pa.size()[0], in_channels, pa.size()[2], pa.size()[3]), requires_grad=False,
                                    dtype=pa.dtype).to(device)
                input[:, 0] = pa[:, 0]
                input[:, 1] = l[:, 0]

            # Forward
            output = model(input)
            running_loss += criterion(output, label).data
            
            # Save outs and labels
            val_preds.append(output.data.cpu().numpy())
            val_true.append(label)
            # Accuracy
            c = (((output > 0.5).to(torch.int) + label.to(torch.int)) == 2).to(torch.float)
            for j in range(trainset.nb_labels):
                class_correct[j] += sum(c[:, j])
                class_total[j] += sum(label[:, j])
            
        
        
        running_loss = running_loss.cpu().detach().numpy().squeeze() / (len(valset) / batch_size)
        val_loss.append(running_loss)
        accuracy = class_correct / class_total
        print(accuracy)
        accuracy = accuracy.cpu().detach().numpy().squeeze()
        val_accuracy.append(accuracy.tolist())
        print('Epoch {0} - Val loss = {1:.5f}'.format(epoch + 1, running_loss))
        print('Epoch {0} - Val accuracy (avg) = {1:.5f}'.format(epoch + 1, accuracy.mean()))

    
        # Save results to df
        try:
            val_preds = np.vstack(val_preds)
            val_true = np.vstack(val_true)
            results_dict = {'accuracy': accuracy_score(val_true, val_preds),
                            'auc': roc_auc_score(val_true, val_preds),
                            'prc': average_precision_score(val_true, val_preds),
                            'loss': running_loss, 'epoch': epoch}
            df = df.append(results_dict, ignore_index=True)
            df.to_csv('./training_results.csv')
        except:
            pass
        
        
        
        with open(join(output_dir, '{}-val_loss.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_loss, f)

        with open(join(output_dir, '{}-val_accuracy.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_loss, f)

        torch.save(model.state_dict(), join(output_dir, '{}-e{}.pt'.format(target, epoch)))

        # Remove all batches weights
        weights_files = glob(join(output_dir, '{}-e{}-i*.pt'.format(target, epoch)))
        for file in weights_files:
            os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('csv_path', type=str)
    parser.add_argument('splits_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)
    args = parser.parse_args()
    df = pd.DataFrame(columns=['accuracy', 'auc', 'prc', 'loss', 'epoch'])
    train(args.data_dir, args.csv_path, args.splits_path, args.output_dir, target=args.target,
          batch_size=args.batch_size, pretrained=args.pretrained, df=df)
