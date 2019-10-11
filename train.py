import argparse
import os
import pickle
from glob import glob
from os.path import join, exists, isfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch import nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, RandomRotation, GaussianNoise, ToPILImage, split_dataset
from models import create_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(data_dir, csv_path, splits_path, output_dir, logdir='./logs', target='pa',
          nb_epoch=100, learning_rate=(1e-4,), batch_size=1, optim='adam',
          dropout=None, min_patients_per_label=50, seed=666, data_augmentation=True,
          model_type='hemis', architecture='densenet121', misc=None):
    assert target in ['pa', 'l', 'joint']

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir.format(seed)
    output_dir = join(logdir, output_dir)
    splits_path = splits_path.format(seed)

    print("Training mode: {}".format(target))

    if not exists(output_dir):
        os.makedirs(output_dir)

    if not exists(splits_path):
        split_dataset(csv_path, splits_path, seed=seed)

    # Find device
    print('Device that will be used is: {0}'.format(DEVICE))

    # Load data
    val_transfo = [Normalize(), ToTensor()]
    if data_augmentation:
        train_transfo = [Normalize(), ToPILImage(), RandomRotation(), ToTensor(), GaussianNoise()]
    else:
        train_transfo = val_transfo

    dset_args = {'datadir': data_dir, 'csvpath': csv_path, 'splitpath': splits_path,
                 'min_patients_per_label': min_patients_per_label, 'flat_dir': misc.flatdir}
    loader_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': misc.threads, 'pin_memory': True}

    trainset = PCXRayDataset(transform=Compose(train_transfo), **dset_args)
    valset = PCXRayDataset(transform=Compose(val_transfo), dataset='val', **dset_args)

    trainloader = DataLoader(trainset, **loader_args)
    valloader = DataLoader(valset, **loader_args)

    print("Number of patients: {} train, {} valid.".format(len(trainset), len(valset)))
    print("Predicting {} labels: {}".format(len(trainset.labels), trainset.labels))
    print(trainset.labels_weights)

    # Load model
    model = create_model(model_type, num_classes=trainset.nb_labels, target=target,
                         architecture=architecture, dropout=dropout, otherargs=misc)
    model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(DEVICE))
    criterion_L = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(DEVICE))
    criterion_PA = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(DEVICE))

    if model_type in ['singletask', 'multitask', 'dualnet'] and len(learning_rate) > 1:
        # each branch has custom learning rate
        optim_params = [{'params': model.frontal_model.parameters(), 'lr': learning_rate[0]},
                        {'params': model.lateral_model.parameters(), 'lr': learning_rate[1]},
                        {'params': model.classifier.parameters(), 'lr': learning_rate[2]}]
    else:
        # one lr for all
        optim_params = [{'params': model.parameters(), 'lr': learning_rate[0]}]

    # Optimizer
    if 'adam' in optim:
        use_amsgrad = 'amsgrad' in optim
        optimizer = Adam(optim_params, weight_decay=misc.weight_decay, amsgrad=use_amsgrad)
    else:
        optimizer = SGD(optim_params, weight_decay=misc.weight_decay, momentum=misc.momentum, nesterov=misc.nesterov)

    scheduler = StepLR(optimizer, step_size=misc.reduce_period, gamma=misc.gamma)  # Used to decay learning rate

    start_epoch = 1
    store_dict = {'train_loss': [], 'val_loss': [], 'val_preds_all': [], 'val_auc': [], 'val_prc': []}
    eval_df = pd.DataFrame(columns=['epoch', 'accuracy', 'auc', 'prc', 'loss'])

    # Resume training if possible
    latest_ckpt_file = join(output_dir, f'{target}-latest.tar')

    if isfile(latest_ckpt_file):
        with torch.load(latest_ckpt_file) as checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        for metric in store_dict.keys():
            with open(join(output_dir, f'{target}-{metric}.pkl'), 'rb') as f:
                store_dict[metric] = pickle.load(f)
        eval_df = pd.read_csv(join(output_dir, f'{target}-metrics.csv'))
        start_epoch = int(eval_df.epoch.iloc[-1])
        print(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, nb_epoch):  # loop over the dataset multiple times
        model.train()

        running_loss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(DEVICE)
        train_preds = []
        train_true = []
        for i, data in enumerate(trainloader, 0):
            if target == 'joint':
                *images, label = data['PA'].to(DEVICE), data['L'].to(DEVICE), data['encoded_labels'].to(DEVICE)
                if model_type == 'stacked':
                    images = torch.cat(images, dim=1)
            else:
                images, label = data[target.upper()].to(DEVICE), data['encoded_labels'].to(DEVICE)

            # Forward
            output = model(images)
            optimizer.zero_grad()
            if model_type == 'multitask':
                joint_logit, frontal_logit, lateral_logit = output
                loss_J = criterion(joint_logit, label)
                loss_PA = criterion_PA(frontal_logit, label) * misc.loss_weights[0]
                loss_L = criterion_L(lateral_logit, label) * misc.loss_weights[1]
                loss = loss_J + loss_L + loss_PA
                output = joint_logit
            else:
                loss = criterion(output, label)

            # Backward
            loss.backward()
            optimizer.step()

            # Save predictions
            train_preds.append(torch.sigmoid(output).detach().cpu().numpy())
            train_true.append(label.detach().cpu().numpy())

            # print statistics
            running_loss += loss.detach().data
            print_every = max(1, len(trainset) // (20 * batch_size))
            if (i + 1) % print_every == 0:
                running_loss = running_loss.cpu().detach().numpy().squeeze() / print_every
                print('[{0}, {1:5}] loss: {2:.5f}'.format(epoch + 1, i + 1, running_loss))
                store_dict['train_loss'].append(running_loss)

                # with open(join(output_dir, '{}-train_loss.pkl'.format(target)), 'wb') as f:
                #     pickle.dump(store_dict['train_loss'], f)
                # torch.save(model.state_dict(), join(output_dir, '{0}-e{1}-i{2}.pt'.format(target, epoch, i + 1)))
                running_loss = torch.zeros(1, requires_grad=False).to(DEVICE)
            del output, images, data

        train_preds = np.vstack(train_preds)
        train_true = np.vstack(train_true)

        model.eval()
        val_true, val_preds, val_runloss = validate(model, dataloader=valloader, loss_fn=criterion, target='joint',
                                                    model_type=model_type, vote_at_test=misc.vote_at_test)
        scheduler.step(epoch=epoch)

        val_runloss = val_runloss.cpu().detach().numpy().squeeze() / (len(valset) / batch_size)
        print('Epoch {0} - Val loss = {1:.5f}'.format(epoch + 1, val_runloss))

        val_auc = roc_auc_score(val_true, val_preds, average=None)
        val_prc = average_precision_score(val_true, val_preds, average=None)

        print("Validation AUC, Train AUC and difference")
        try:
            train_auc = roc_auc_score(train_true, train_preds, average=None)
        except:
            print('Error in calculating train AUC')
            train_auc = np.zeros_like(val_auc)

        diff_train_val = val_auc - train_auc
        diff_train_val = np.stack([val_auc, train_auc, diff_train_val], axis=-1)
        print(diff_train_val.round(4))
        print()

        store_dict['val_prc'].append(val_prc)
        store_dict['val_preds_all'].append(val_preds)
        store_dict['val_auc'].append(val_auc)
        store_dict['val_loss'].append(val_runloss)

        for metric in store_dict.keys():
            with open(join(output_dir, '{}-{}.pkl'.format(target, metric)), 'wb') as f:
                pickle.dump(store_dict[metric], f)

        metrics = {'epoch': epoch + 1,
                   'accuracy': accuracy_score(val_true, np.where(val_preds > 0.5, 1, 0)),
                   'auc': roc_auc_score(val_true, val_preds, average='weighted'),
                   'prc': average_precision_score(val_true, val_preds, average='weighted'),
                   'loss': running_loss}

        eval_df = eval_df.append(metrics, ignore_index=True)
        print(metrics)
        eval_df.to_csv(join(output_dir, '{}-metrics.csv'.format(target)))

        _states = {'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'scheduler_state_dict': scheduler.state_dict()}
        torch.save(_states, latest_ckpt_file)
        torch.save(model.state_dict(), join(output_dir, '{}-e{}.pt'.format(target, epoch)))

        # # Remove all batches weights
        # weights_files = glob(join(output_dir, '{}-e{}-i*.pt'.format(target, epoch)))
        # for file in weights_files:
        #     os.remove(file)
        #


def validate(model, dataloader, loss_fn, target='joint', model_type=None, vote_at_test=False):
    with torch.no_grad():
        runningloss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(DEVICE)
        y_preds, y_true = [], []
        for data in dataloader:
            if target == 'joint':
                *images, label = data['PA'].to(DEVICE), data['L'].to(DEVICE), data['encoded_labels'].to(DEVICE)
                if model_type == 'stacked':
                    images = torch.cat(images, dim=1)
            else:
                images, label = data[target.upper()].to(DEVICE), data['encoded_labels'].to(DEVICE)

            # Forward
            output = model(images)
            if model_type == 'multitask':
                if vote_at_test:
                    output = torch.stack(output, dim=1).mean(dim=1)
                else:
                    output = output[0]

            runningloss += loss_fn(output, label).mean().detach().data

            # Save predictions
            y_preds.append(torch.sigmoid(output).detach().cpu().numpy())
            y_true.append(label.detach().cpu().numpy())
            del output, images, data

    y_true = np.vstack(y_true)
    y_preds = np.vstack(y_preds)
    return y_true, y_preds, runningloss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')

    # Paths
    parser.add_argument('data_dir', type=str)
    parser.add_argument('csv_path', type=str)
    parser.add_argument('splits_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--exp_name', type=str, default=None)

    # Model params
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--model-type', type=str, default='hemis',
                        help="Which joint model: must be one of ['multitask', 'dualnet', 'stacked', 'hemis', 'concat']")
    parser.add_argument('--vote-at-test', action='store_true')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=[0.0001], nargs='*')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--gamma', type=int, default=0.5)
    parser.add_argument('--reduce_period', type=int, default=20)

    # Dataset params
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--min_patients', type=int, default=50)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--threads', type=int, default=1)

    # Other optional arguments
    parser.add_argument('--merge', type=int, default=3,
                        help='For Hemis and HemisConcat. Merge modalities after N blocks')
    parser.add_argument('--drop-view-prob', type=float, default=0.0,
                        help='For joint. Drop either view with p/2 and keep both views with 1-p')
    parser.add_argument('--mt-combine-at', dest='combine', type=str, default='prepool',
                        help='For Multitask. Combine both views before or after pooling')
    parser.add_argument('--mt-join', dest='join', type=str, default='concat',
                        help='For Multitask. Combine views how? Valid options - concat, max, mean')
    parser.add_argument('--loss-weights', type=float, default=(0.3, 0.3), nargs=2,
                        help='For Multitask. Loss weights for regularizing loss. 1st is for PA, 2nd for L')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--flatdir', action='store_false')
    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    if args.exp_name:
        args.output_dir = args.output_dir + "-" + args.exp_name

    print(args)
    train(args.data_dir, args.csv_path, args.splits_path, args.output_dir, logdir=args.logdir,
          target=args.target, batch_size=args.batch_size, nb_epoch=args.epochs,
          learning_rate=args.learning_rate, min_patients_per_label=args.min_patients, dropout=args.dropout,
          seed=args.seed, model_type=args.model_type, architecture=args.arch, misc=args)
