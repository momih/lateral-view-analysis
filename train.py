from glob import glob
import os
import pickle

import numpy as np

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch import nn
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor
from densenet import DenseNet, add_dropout


torch.manual_seed(42)
np.random.seed(42)


def train(data_dir, csv_path, split_path, target='pa', nb_epoch=100, learning_rate=1e-4, batch_size=1, dropout=True):
    assert target in ['pa', 'l', 'joint']

    # Load data
    preprocessing = Compose([Normalize(), ToTensor()])
    trainset = PCXRayDataset(data_dir, csv_path, split_path, transform=preprocessing)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    valset = PCXRayDataset(data_dir, csv_path, split_path, transform=preprocessing, dataset='val')
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("{0} patients in training set.".format(len(trainset)))
    print("{0} patients in validation set.".format(len(valset)))

    # Load model
    model = DenseNet(out_features=trainset.nb_labels)

    # Add dropout
    if dropout:
        model = add_dropout(model, p=0.2)

    criterion = nn.BCELoss(size_average=True)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Used to decay learning rate

    # Resume training if possible
    start_epoch = 0
    start_batch = 0
    train_loss = []
    val_loss = []
    weights_files = glob('models/baseline-e*.pt')  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array([int(w[len('models/baseline-e'):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max() + 1
        weights_files = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()]

        # Find most recent batch
        if len(weights_files) > 1:
            batches = np.array([int(w[len('models/baseline-e'):-len('.pt')].split('i')[1]) for w in weights_files])
            start_batch = batches.max()
            weights_file = weights_files[np.argmax(batches)]
            start_epoch -= 1
        else:
            weights_file = weights_files[0]
        model.load_state_dict(torch.load(weights_file))

        with open('./models/train_loss.pkl', 'rb') as f:
            train_loss = pickle.load(f)

        with open('./models/val_loss.pkl', 'rb') as f:
            val_loss = pickle.load(f)

        print("Resuming training at epoch {0}.".format(start_epoch))
        print("Weights loaded: {0}".format(weights_file))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device that will be used is: {0}'.format(device))

    model.to(device)

    # Training loop
    for epoch in range(start_epoch, nb_epoch):  # loop over the dataset multiple times
        scheduler.step()

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            model.train()

            # Skip to current batch
            if epoch == start_epoch and i < start_batch:
                continue

            if target is 'pa':
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)
            elif target is 'l':
                input, label = data['L'].to(device), data['encoded_labels'].to(device)
            else:
                # TODO
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)

            # Forward
            output = model(input)[-1]
            optimizer.zero_grad()
            loss = criterion(output, label)

            # Backward
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss
            print_every = max(1, len(trainset) // (20 * batch_size))
            if (i + 1) % print_every == 0:
                running_loss = running_loss / print_every
                print('[{0}, {1:5}] loss: {2:.5f}'.format(epoch + 1, i + 1, running_loss))
                train_loss.append(running_loss)

                with open('./models/train_loss.pkl', 'wb') as f:
                    pickle.dump(train_loss, f)
                torch.save(model.state_dict(), 'models/baseline-e{0}-i{1}.pt'.format(epoch, i + 1))
                running_loss = 0.0

        running_loss = 0.0
        for i, data in enumerate(valloader, 0):
            model.eval()

            if target is 'pa':
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)
            elif target is 'l':
                input, label = data['L'].to(device), data['encoded_labels'].to(device)
            else:
                # TODO
                input, label = data['PA'].to(device), data['encoded_labels'].to(device)

            # Forward
            output = model(input)[-1]
            running_loss += criterion(output, label)

        running_loss /= (len(valset) / batch_size)
        val_loss.append(running_loss)
        print('Epoch {} - Val loss = {}'.format(epoch + 1, running_loss))

        with open('./models/val_loss.pkl', 'wb') as f:
            pickle.dump(val_loss, f)

        torch.save(model.state_dict(), 'models/baseline-e{0}.pt'.format(epoch))

        # Remove all batches weights
        weights_files = glob('models/baseline-e{0}-i*.pt'.format(epoch))
        for file in weights_files:
            os.remove(file)
            os.remove(file.replace('encoder', 'decoder'))


if __name__ == "__main__":
    data_dir = './data/processed/'
    csv_path = './data/cxr8_joint_cohort_data.csv'
    split_file = './models/data_split.pkl'

    train(data_dir, csv_path, split_file)
