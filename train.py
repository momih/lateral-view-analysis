#!/usr/bin/env python
import argparse
import logging
import os
from glob import glob
from os.path import join, exists, isfile

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, RandomRotation, RandomTranslate, GaussianNoise, ToPILImage, \
    split_dataset
from evaluate import ModelEvaluator, get_model_preds
from models import create_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def create_opt_and_sched(optim, params, lr, other_args):
    # Optimizer
    if 'adam' in optim:
        use_amsgrad = 'amsgrad' in optim
        opt = Adam
        if 'w' in optim:
            try:
                opt = torch.optim.AdamW
                print('Using AdamW')
            except AttributeError:
                pass
        optimizer = opt(params, lr=lr, weight_decay=other_args.weight_decay, amsgrad=use_amsgrad)

    elif optim == 'rmsprop':
        optimizer = RMSprop(params, lr=lr, weight_decay=other_args.weight_decay, momentum=other_args.momentum)

    else:
        optimizer = SGD(params, lr=lr, weight_decay=other_args.weight_decay,
                        momentum=other_args.momentum, nesterov=other_args.nesterov)

    if 'reduce' in other_args.sched:
        scheduler = ReduceLROnPlateau(optimizer, factor=other_args.gamma, patience=other_args.reduce_period,
                                      verbose=True, threshold=0.001)
    else:
        scheduler = StepLR(optimizer, step_size=other_args.reduce_period, gamma=other_args.gamma)

    return optimizer, scheduler


def train(data_dir, csv_path, splits_path, output_dir, target='pa', nb_epoch=100, lr=(1e-4,), batch_size=1,
          optim='adam', dropout=None, min_patients_per_label=50, seed=666, data_augmentation=True, model_type='hemis',
          architecture='densenet121', misc=None):
    assert target in ['pa', 'l', 'joint']

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir.format(seed)
    splits_path = splits_path.format(seed)

    logger.info(f"Training mode: {target}")

    if not exists(output_dir):
        os.makedirs(output_dir)

    if not exists(splits_path):
        split_dataset(csv_path, splits_path, seed=seed)

    # Find device
    logger.info(f'Device that will be used is: {DEVICE}')

    # Load data
    val_transfo = [Normalize(), ToTensor()]
    if data_augmentation:
        train_transfo = [Normalize(), ToPILImage()]

        if 'rotation' in misc.transforms:
            train_transfo.append(RandomRotation(degrees=misc.rotation_degrees))

        if 'translation' in misc.transforms:
            train_transfo.append(RandomTranslate(translate=misc.translate))

        train_transfo.append(ToTensor())

        if 'noise' in misc.transforms:
            train_transfo.append(GaussianNoise())
    else:
        train_transfo = val_transfo

    dset_args = {'datadir': data_dir, 'csvpath': csv_path, 'splitpath': splits_path,
                 'max_label_weight': misc.max_label_weight, 'min_patients_per_label': min_patients_per_label,
                 'flat_dir': misc.flatdir}
    loader_args = {'batch_size': batch_size, 'shuffle': True, 'num_workers': misc.threads, 'pin_memory': True}

    trainset = PCXRayDataset(transform=Compose(train_transfo), **dset_args)
    valset = PCXRayDataset(transform=Compose(val_transfo), dataset='val', **dset_args)

    trainloader = DataLoader(trainset, **loader_args)
    valloader = DataLoader(valset, **loader_args)
    n_pts = f"{len(trainset)} train,"

    if misc.use_extended:
        ext_args = dset_args.copy()
        ext_args['splitpath'] = None
        ext_args['csvpath'] = misc.csv_path_ext

        extset = PCXRayDataset(transform=Compose(train_transfo), mode='pa_only',
                               use_labels=trainset.labels, **ext_args)
        extset.labels_count = trainset.labels_count
        extset.labels_weights = trainset.labels_weights
        extloader = DataLoader(extset, **loader_args)

        n_pts += f" {len(extset)} ext_train,"

    logger.info(f"Number of patients: {n_pts} {len(valset)} valid.")
    logger.info(f"Predicting {len(trainset.labels)} labels: \n{trainset.labels}")
    logger.info(trainset.labels_weights)

    # Load model
    model = create_model(model_type, num_classes=trainset.nb_labels, target=target,
                         architecture=architecture, dropout=dropout, otherargs=misc)
    model.to(DEVICE)
    logger.info(f'Created {model_type} model')

    evaluator = ModelEvaluator(output_dir=output_dir, target=target, logger=logger)

    criterion = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(DEVICE))
    loss_weights = [1.0] + list(misc.loss_weights)

    if len(misc.mt_task_prob) == 1:
        _mt_task_prob = misc.mt_task_prob[0]
        task_prob = [1 - _mt_task_prob, _mt_task_prob / 2., _mt_task_prob / 2.]
    else:
        _pa_prob, _l_prob = misc.mt_task_prob
        _jt_prob = 1 - (_pa_prob + _l_prob)
        task_prob = [_jt_prob, _pa_prob, _l_prob]

    if model_type in ['singletask', 'multitask', 'dualnet'] and len(lr) > 1:
        # each branch has custom learning rate
        optim_params = [{'params': model.frontal_model.parameters(), 'lr': lr[0]},
                        {'params': model.lateral_model.parameters(), 'lr': lr[1]},
                        {'params': model.classifier.parameters(), 'lr': lr[2]}]
    else:
        # one lr for all
        optim_params = [{'params': model.parameters(), 'lr': lr[0]}]

    if misc.learn_loss_coeffs:
        temperature = torch.ones(size=(3,), requires_grad=True, device=DEVICE).float()
        temperature_lr = lr[-1] if len(lr) > 3 else lr[0]
        optim_params.append({'params': temperature, 'lr': temperature_lr})

    # Optimizer
    optimizer, scheduler = create_opt_and_sched(optim=optim, params=optim_params, lr=lr[0], other_args=misc)
    start_epoch = 1

    # Resume training if possible
    latest_ckpt_file = join(output_dir, f'{target}-latest.tar')
    if isfile(latest_ckpt_file):
        checkpoint = torch.load(latest_ckpt_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        del checkpoint

        evaluator.load_saved()
        start_epoch = int(evaluator.eval_df.epoch.iloc[-1])
        logger.info(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, nb_epoch + 1):  # loop over the dataset multiple times
        model.train()
        running_loss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(DEVICE)
        train_preds, train_true = [], []
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
                # order of returned logits is joint, frontal, lateral
                if misc.learn_loss_coeffs:
                    loss_weights = temperature.pow(-2)

                all_task_losses, weighted_task_losses = [], []
                for idx, _logit in enumerate(output):
                    task_loss = criterion(_logit, label)
                    all_task_losses.append(task_loss)
                    weighted_task_losses.append(task_loss * loss_weights[idx])

                losses_dict = {0: sum(weighted_task_losses), 1: all_task_losses[1], 2: all_task_losses[2]}
                select = np.random.choice([0, 1, 2], p=task_prob)
                loss = losses_dict[select]  # mixing this temp seems bad

                if misc.learn_loss_coeffs:
                    loss += temperature.log().sum()

                output = output[0]
            else:
                loss = criterion(output, label)

            # Backward
            loss.backward()
            optimizer.step()

            # Save predictions
            train_preds.append(torch.sigmoid(output).detach().cpu().numpy())
            train_true.append(label.detach().cpu().numpy())

            # print statistics
            running_loss += loss.detach()
            print_every = max(1, len(trainset) // (20 * batch_size))
            if (i + 1) % print_every == 0:
                running_loss = running_loss.cpu().detach().numpy().squeeze() / print_every
                logger.info('[{0}, {1:5}] loss: {2:.5f}'.format(epoch, i + 1, running_loss))
                evaluator.store_dict['train_loss'].append(running_loss)
                running_loss = torch.zeros(1, requires_grad=False).to(DEVICE)
            del output, images, data

        if misc.use_extended:
            # Train with only PA images from extended dataset
            for i, data in enumerate(extloader, 0):
                if target == 'joint':
                    *images, label = data['PA'].to(DEVICE), data['L'].to(DEVICE), data['encoded_labels'].to(DEVICE)
                else:
                    images, label = data[target.upper()].to(DEVICE), data['encoded_labels'].to(DEVICE)

                # Forward
                output = model(images)
                optimizer.zero_grad()
                if model_type == 'multitask':
                    # only use PA loss
                    output = output[1]

                loss = criterion(output, label)

                # Backward
                loss.backward()
                optimizer.step()

                # Save predictions
                train_preds.append(torch.sigmoid(output).detach().cpu().numpy())
                train_true.append(label.detach().cpu().numpy())

                # print statistics
                running_loss += loss.detach()
                print_every = max(1, len(trainset) // (20 * batch_size))
                if (i + 1) % print_every == 0:
                    running_loss = running_loss.cpu().detach().numpy().squeeze() / print_every
                    logger.info('[{0}, {1:5}] Extended dataset loss: {2:.5f}'.format(epoch, i + 1, running_loss))
                    evaluator.store_dict['train_loss'].append(running_loss)
                    running_loss = torch.zeros(1, requires_grad=False).to(DEVICE)
                del output, images, data

        train_preds = np.vstack(train_preds)
        train_true = np.vstack(train_true)

        model.eval()
        val_true, val_preds, val_runloss = get_model_preds(model, dataloader=valloader, loss_fn=criterion,
                                                           target=target, model_type=model_type,
                                                           vote_at_test=misc.vote_at_test)

        val_runloss /= (len(valset) / batch_size)
        logger.info(f'Epoch {epoch} - Val loss = {val_runloss:.5f}')
        val_auc, _ = evaluator.evaluate_and_save(val_true, val_preds, epoch=epoch,
                                                 train_true=train_true, train_preds=train_preds,
                                                 runloss=val_runloss)

        if 'reduce' in misc.sched:
            scheduler.step(metrics=val_auc, epoch=epoch)
        else:
            scheduler.step(epoch=epoch)

        _states = {'model_state_dict': model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'scheduler_state_dict': scheduler.state_dict()}
        torch.save(_states, latest_ckpt_file)
        torch.save(model.state_dict(), join(output_dir, '{}-e{}.pt'.format(target, epoch)))

        # Remove all batches weights
        weights_files = glob(join(output_dir, '{}-e{}-i*.pt'.format(target, epoch)))
        for file in weights_files:
            os.remove(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')

    # Paths
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--splits_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)

    # Model params
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--model-type', type=str, default='hemis',
                        help="Which joint model to pick: must be one of "
                             "['multitask', 'dualnet', 'stacked', 'hemis', 'concat']")
    parser.add_argument('--vote-at-test', action='store_true')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=[0.0001], nargs='*')
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--sched', default='steplr')
    parser.add_argument('--reduce_period', type=int, default=20)

    # Dataset params
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--min_patients', type=int, default=50)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--max_label_weight', default=5.0, type=float)
    parser.add_argument('--use-extended', action='store_true')
    parser.add_argument('--csv_path_ext', default=None)

    # Data augmentation options
    parser.add_argument('--data-augmentation', type=bool, default=True)
    parser.add_argument('--transforms', default=['rotation', 'translation', 'noise'], nargs='*')
    parser.add_argument('--rotation-degrees', type=int, default=5)
    parser.add_argument('--translate', type=float, default=None, nargs=2,
                        help="tuple of 2 fractions for width and height")

    # Other optional arguments
    parser.add_argument('--merge', type=int, default=3,
                        help='For Hemis and HemisConcat. Merge modalities after N blocks')
    parser.add_argument('--drop-view-prob', type=float, default=0.0,
                        help='For joint. Drop either view with p/2 and keep both views with 1-p. '
                             'Disabled for multitask')
    parser.add_argument('--mt-task-prob', type=float, default=[0.0], nargs='*',
                        help='Curriculum learning probs for multitask. For PA and L resp'
                             'If only one arg, then drop either task with p/2 and keep both views with 1-p')
    parser.add_argument('--mt-combine-at', dest='combine', type=str, default='prepool',
                        help='For Multitask. Combine both views before or after pooling')
    parser.add_argument('--mt-join', dest='join', type=str, default='concat',
                        help='For Multitask. Combine views how? Valid options - concat, max, mean')

    parser.add_argument('--learn-loss-coeffs', action='store_true', help='Learn the loss weights')
    parser.add_argument('--loss-weights', type=float, default=[0.3, 0.3], nargs=2,
                        help='For Multitask. Loss weights for regularizing loss. 1st is for PA, 2nd for L')
    parser.add_argument('--nesterov', action='store_true')

    parser.add_argument('--momentum', default=0.0, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)
    parser.add_argument('--flatdir', action='store_false')
    parser.add_argument('--print-every', default=20, type=int)

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    if args.exp_name:
        args.output_dir = args.output_dir + "-" + args.exp_name

    if args.dropout >= 1:
        args.dropout /= 10.

    if args.data_dir == "CLUSTER":
        args.data_dir = os.environ.get('DATADIR')

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
    train(data_dir=args.data_dir, csv_path=args.csv_path, splits_path=args.splits_path,
          output_dir=args.output_dir, target=args.target, nb_epoch=args.epochs, lr=args.learning_rate,
          batch_size=args.batch_size, optim=args.optim, dropout=args.dropout,
          min_patients_per_label=args.min_patients, seed=args.seed, model_type=args.model_type,
          architecture=args.arch, data_augmentation=args.data_augmentation, misc=args)
