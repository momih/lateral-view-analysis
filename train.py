#!/usr/bin/env python
import argparse
import logging
import os
import pickle
from glob import glob
from os.path import join, exists

import mlflow
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
from models import DenseNet, HeMIS, HeMISConcat, FrontalLateralMultiTask, ResNet
from models import add_dropout, get_densenet_params, get_resnet_params

from orion.client import report_results


logger = logging.getLogger(__name__)


def train(data_dir, csv_path, splits_path, output_dir, logdir='./logs', target='pa',
          nb_epoch=100, learning_rate=1e-4, batch_size=1, optim='adam',
          dropout=None, pretrained=False, min_patients_per_label=50, seed=666, data_augmentation=True,
          model_type='hemis', architecture='densenet121', other_args=None, threads=0):
    assert target in ['pa', 'l', 'joint']

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir.format(seed)
    output_dir = join(logdir, output_dir)
    splits_path = splits_path.format(seed)

    logger.info("Training mode: {}".format(target))

    if not exists(output_dir):
        os.makedirs(output_dir)

    if not exists(splits_path):
        split_dataset(csv_path, splits_path, seed=seed)

    # Find device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info('Device that will be used is: {0}'.format(device))

    # Logging hparams
    mlflow.log_param('model_type', model_type)
    mlflow.log_param('target', target)
    mlflow.log_param('seed', seed)
    mlflow.log_param('optimizer', optim)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('gamma', other_args.gamma)
    mlflow.log_param('reduce_period', other_args.reduce_period)
    mlflow.log_param('dropout', dropout)

    # Load data
    val_transfo = [Normalize(), ToTensor()]
    if data_augmentation:
        train_transfo = [Normalize(), ToPILImage(), RandomRotation(), ToTensor(), GaussianNoise()]
    else:
        train_transfo = val_transfo

    trainset = PCXRayDataset(data_dir, csv_path, splits_path, transform=Compose(train_transfo), pretrained=pretrained,
                             min_patients_per_label=min_patients_per_label, flat_dir=other_args.flatdir)
    
    logger.info("predicting {} labels: {}".format(len(trainset.labels), trainset.labels))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=threads, pin_memory=True)

    valset = PCXRayDataset(data_dir, csv_path, splits_path, transform=Compose(val_transfo), dataset='val',
                           pretrained=pretrained, min_patients_per_label=min_patients_per_label, flat_dir=other_args.flatdir)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=True,
                           num_workers=threads, pin_memory=True)

    logger.info("{0} patients in training set.".format(len(trainset)))
    logger.info("{0} patients in validation set.".format(len(valset)))

    # Load model
    in_channels = 3 if pretrained else 1

    if target == 'joint':
        if model_type in ['singletask', 'multitask', 'dualnet']:
            joint_only = model_type != 'multitask'
            model = FrontalLateralMultiTask(num_classes=trainset.nb_labels, combine_at=other_args.combine,
                                            join_how=other_args.join, drop_view_prob=other_args.drop_view_prob,
                                            joint_only=joint_only, architecture=architecture)
        elif model_type == 'stacked':
            modelparams = get_densenet_params(architecture)
            model = DenseNet(num_classes=trainset.nb_labels, in_channels=2, **modelparams)
        
        elif model_type == 'concat':
            model = HeMISConcat(num_classes=trainset.nb_labels, in_channels=1, merge_at=other_args.merge,
                                drop_view_prob=other_args.drop_view_prob)
        
        else:  # Default HeMIS
            model = HeMIS(num_classes=trainset.nb_labels, in_channels=1, merge_at=other_args.merge,
                          drop_view_prob=other_args.drop_view_prob)
            model_type = 'hemis'
    else:
        if 'resnet' in architecture:
            modelparams = get_resnet_params(architecture)
            model = ResNet(num_classes=trainset.nb_labels, in_channels=in_channels, **modelparams)
            model_type = 'resnet'
        else:
            # Use DenseNet by default
            modelparams = get_densenet_params(architecture)
            model = DenseNet(num_classes=trainset.nb_labels, in_channels=in_channels, **modelparams)
            model_type = 'densenet'
    
    logger.info('Created {} model'.format(model_type))
    # Add dropout
    if dropout:
        model = add_dropout(model, p=dropout, model=model_type)

    logger.info(trainset.labels_weights)

    criterion = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(device))

    if model_type == 'multitask':
        criterion_L = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(device))
        criterion_PA = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(device))

    # Optimizer
    if 'adam' in optim:
        use_amsgrad = 'amsgrad' in optim
        optimizer = Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
                         eps=1e-08, weight_decay=1e-5, amsgrad=use_amsgrad)
    else:
        optimizer = SGD(model.parameters(), lr=learning_rate, weight_decay=1e-5,
                        momentum=other_args.momentum, nesterov=other_args.nesterov)

    scheduler = StepLR(optimizer, step_size=other_args.reduce_period, gamma=other_args.gamma)
    # Used to decay learning rate

    # Resume training if possible
    start_epoch = 0
    start_batch = 0
    train_loss = []
    val_loss = []
    val_preds_all = []
    val_auc = []
    val_prc = []
    metrics_df = pd.DataFrame(columns=['accuracy', 'auc', 'prc', 'loss', 'epoch', 'error'])
    
    weights_files = glob(join(output_dir, '{}-e*.pt'.format(target)))  # Find all weights files
    if len(weights_files):
        # Find most recent epoch
        epochs = np.array(
            [int(w[len(join(output_dir, '{}-e'.format(target))):-len('.pt')].split('-')[0]) for w in weights_files])
        start_epoch = epochs.max() + 1
        weights_files = [weights_files[i] for i in np.argwhere(epochs == np.amax(epochs)).flatten()]

        # Find most recent batch
        if len(weights_files) > 1:
            batches = np.array(
                [int(w[len(join(output_dir, '{}-e'.format(target))):-len('.pt')].split('i')[1]) for w in weights_files])
            start_batch = batches.max()
            weights_file = weights_files[np.argmax(batches)]
            start_epoch -= 1
        else:
            weights_file = weights_files[0]
        model.load_state_dict(torch.load(weights_file))

        with open(join(output_dir, '{}-train_loss.pkl'.format(target)), 'rb') as f:
            train_loss = pickle.load(f)

        with open(join(output_dir, '{}-val_preds.pkl'.format(target)), 'rb') as f:
            val_preds_all = pickle.load(f)

        with open(join(output_dir, '{}-val_loss.pkl'.format(target)), 'rb') as f:
            val_loss = pickle.load(f)

        with open(join(output_dir, '{}-val_auc.pkl'.format(target)), 'rb') as f:
            val_auc = pickle.load(f)

        with open(join(output_dir, '{}-val_prc.pkl'.format(target)), 'rb') as f:
            val_prc = pickle.load(f)

        metrics_df = pd.read_csv(join(output_dir, '{}-metrics.csv'.format(target)),
                                 usecols=['accuracy', 'auc', 'prc', 'loss', 'epoch', 'error'], low_memory=False)

        logger.info("Resuming training at epoch {0}.".format(start_epoch))
        logger.info("Weights loaded: {0}".format(weights_file))

    model.to(device)

    # Training loop
    for epoch in range(start_epoch, nb_epoch):  # loop over the dataset multiple times
        model.train()

        running_loss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(device)
        train_preds = []
        train_true = []
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
                input = [pa, l]
                if model_type == 'stacked':
                    input = torch.cat(input, dim=1)

            # Forward
            output = model(input)
            optimizer.zero_grad()
            if model_type == 'multitask':
                joint_logit, frontal_logit, lateral_logit = output
                loss_J = criterion(joint_logit, label)
                loss_PA = criterion_PA(frontal_logit, label) * other_args.loss_weights[0]
                loss_L = criterion_L(lateral_logit, label) * other_args.loss_weights[1]
                loss = loss_J + loss_L + loss_PA

                output = joint_logit
            else:
                loss = criterion(output, label)
            # loss = (loss * sample_weights / sample_weights.sum()).sum()

            # Backward
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Save predictions
            train_preds.append(torch.sigmoid(output).detach().data.cpu().numpy())
            train_true.append(label.data.detach().cpu().numpy())

            # print statistics
            running_loss += loss.detach().data
            print_every = max(1, len(trainset) // (20 * batch_size))
            if (i + 1) % print_every == 0:
                running_loss = running_loss.cpu().detach().numpy().squeeze() / print_every
                logger.info('[{0}, {1:5}] loss: {2:.5f}'.format(epoch + 1, i + 1, running_loss))
                train_loss.append(running_loss)

                with open(join(output_dir, '{}-train_loss.pkl'.format(target)), 'wb') as f:
                    pickle.dump(train_loss, f)
                torch.save(model.state_dict(), join(output_dir, '{0}-e{1}-i{2}.pt'.format(target, epoch, i + 1)))
                running_loss = torch.zeros(1, requires_grad=False).to(device)
            del output
            del input
            del data

        train_preds = np.vstack(train_preds)
        train_true = np.vstack(train_true)

        model.eval()
        
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
                input = [pa, l]
                if model_type == 'stacked':
                    input = torch.cat(input, dim=1)
                
            # Forward
            output = model(input)
            if model_type == 'multitask':
                if other_args.vote_at_test:
                    output = torch.stack(output, dim=1).mean(dim=1)
                else:
                    output = output[0]

            running_loss += criterion(output, label).mean().detach().data

            # Save predictions
            val_preds.append(torch.sigmoid(output).data.cpu().numpy())
            val_true.append(label.data.cpu().detach().numpy())
            del output
            del input
            del data

        running_loss = running_loss.cpu().detach().numpy().squeeze() / (len(valset) / batch_size)
        val_loss.append(running_loss)
        logger.info('Epoch {0} - Val loss = {1:.5f}'.format(epoch + 1, running_loss))

        val_preds = np.vstack(val_preds)
        val_true = np.vstack(val_true)
        val_preds_all.append(val_preds)
        auc = roc_auc_score(val_true, val_preds, average=None)
        val_auc.append(auc)
        
        # TODO add options to print
        logger.info("Validation AUC, Train AUC and difference")
        try:
            train_auc = roc_auc_score(train_true, train_preds, average=None)
        except:
            logger.error('Error in calculating train AUC')
            train_auc = np.zeros_like(auc)

        diff_train_val = auc - train_auc
        diff_train_val = np.stack([auc, train_auc, diff_train_val], axis=-1)
        logger.info(diff_train_val.round(4))

        prc = average_precision_score(val_true, val_preds, average=None)
        val_prc.append(prc)

        metrics = {'accuracy': accuracy_score(val_true, np.where(val_preds > 0.5, 1, 0)),
                   'auc': roc_auc_score(val_true, val_preds, average='weighted'),
                   'prc': average_precision_score(val_true, val_preds, average='weighted'),
                   'loss': running_loss,
                   'epoch': epoch + 1}
        metrics_df = metrics_df.append(metrics, ignore_index=True)
        logger.info(metrics)

        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=epoch + 1)

        with open(join(output_dir, '{}-val_preds.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_preds_all, f)

        with open(join(output_dir, '{}-val_loss.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_loss, f)

        with open(join(output_dir, '{}-val_auc.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_auc, f)

        with open(join(output_dir, '{}-val_prc.pkl'.format(target)), 'wb') as f:
            pickle.dump(val_prc, f)

        metrics_df.to_csv(join(output_dir, '{}-metrics.csv'.format(target)))

        torch.save(model.state_dict(), join(output_dir, '{}-e{}.pt'.format(target, epoch)))

        # Remove all batches weights
        weights_files = glob(join(output_dir, '{}-e{}-i*.pt'.format(target, epoch)))
        for file in weights_files:
            os.remove(file)

    return metrics_df['loss'][-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')

    # Paths
    parser.add_argument('data_dir', type=str)
    parser.add_argument('csv_path', type=str)
    parser.add_argument('splits_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--logdir', type=str, default='./logs')
    parser.add_argument('--exp_name', type=str, default=None)

    # Model params
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--model-type', type=str, default='hemis', 
                        help="Which joint model to pick: must be one of "
                             "['multitask', 'dualnet', 'stacked', 'hemis', 'concat']")
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--vote-at-test', action='store_true')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--reduce_period', type=int, default=20)

    # Dataset params
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--min_patients', type=int, default=50)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--threads', type=int, default=0)

    # Other optional arguments
    parser.add_argument('--merge', type=int, default=3,
                        help='For Hemis and HemisConcat. Merge modalities after N blocks')
    parser.add_argument('--drop-view-prob', type=float, default=0.0,
                        help='For Hemis, HemisConcat and Multitask. Drop either '
                             'view with prob/2 and keep both views with 1-prob')
    parser.add_argument('--mt-combine-at', dest='combine', type=str, default='prepool',
                        help='For Multitask. Combine both views before or after pooling')
    parser.add_argument('--mt-join', dest='join', type=str, default='concat',
                        help='For Multitask. Combine views how? Valid options - concat, max, mean')
    parser.add_argument('--loss-weights', type=float, default=(0.3, 0.3), nargs=2,
                        help='For Multitask. Loss weights for regularizing loss. 1st is for PA, 2nd for L')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--flatdir', action='store_false')

    args = parser.parse_args()
    np.set_printoptions(suppress=True, precision=4)

    if args.exp_name:
        args.output_dir = args.output_dir + "-" + args.exp_name

    if args.dropout >= 1:
        args.dropout /= 10.

    if args.data_dir == "CLUSTER":
        args.data_dir = os.environ.get('DATADIR')

    mlflow.set_experiment('lateral-view-analysis')
    mlflow.start_run(run_name=f'{args.model_type}-run{args.exp_name}')

    logging.basicConfig(level=logging.INFO)

    # will log to a file if provided
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)
    
    logger.info(args)
    val_loss = train(args.data_dir, args.csv_path, args.splits_path, args.output_dir, logdir=args.logdir,
                     target=args.target, batch_size=args.batch_size, nb_epoch=args.epochs, pretrained=args.pretrained,
                     learning_rate=args.learning_rate, min_patients_per_label=args.min_patients, dropout=args.dropout,
                     seed=args.seed, model_type=args.model_type, architecture=args.arch,
                     other_args=args, threads=args.threads)

    report_results([dict(
        name='val_loss',
        type='objective',
        value=val_loss)])
