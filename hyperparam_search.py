#!/usr/bin/env python
import argparse
import json
import logging
import os
from glob import glob
from os.path import join, exists, isfile

import mlflow
import numpy as np
import torch
from orion.client import report_results
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, RandomRotation, RandomTranslate, GaussianNoise, ToPILImage, \
    split_dataset
from evaluate import ModelEvaluator, get_model_preds
from models import create_model
from train import create_opt_and_sched

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logger = logging.getLogger(__name__)


def train(data_dir, csv_path, splits_path, output_dir, target='pa', nb_epoch=100, learning_rate=(1e-4,), batch_size=1,
          dropout=None, optim='adam', min_patients_per_label=50, seed=666, data_augmentation=True, model_type='hemis',
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

    # Logging hparams
    mlflow.log_param('model_type', model_type)
    mlflow.log_param('architecture', architecture)
    mlflow.log_param('target', target)
    mlflow.log_param('seed', seed)
    mlflow.log_param('optimizer', optim)
    for i, lr in enumerate(learning_rate):
        mlflow.log_param(f'learning_rate_{i}', lr)
    mlflow.log_param('gamma', misc.gamma)
    mlflow.log_param('reduce_period', misc.reduce_period)
    mlflow.log_param('dropout', dropout)
    mlflow.log_param('max_label_weight', misc.max_label_weight)

    if model_type == 'hemis':
        mlflow.log_param('drop-view-prob', misc.drop_view_prob)

    if model_type == 'multitask':
        mlflow.log_param('mt-task-prob', misc.mt_task_prob)
        mlflow.log_param('mt-join', misc.join)

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

    logger.info("Number of patients: {} train, {} valid.".format(len(trainset), len(valset)))
    logger.info("Predicting {} labels: {}".format(len(trainset.labels), trainset.labels))
    logger.info(trainset.labels_weights)

    # Load model
    model = create_model(model_type, num_classes=trainset.nb_labels, target=target,
                         architecture=architecture, dropout=dropout, otherargs=misc)
    model.to(DEVICE)
    logger.info(f'Created {model_type} model')

    evaluator = ModelEvaluator(output_dir=output_dir, target=target, logger=logger)

    criterion = nn.BCEWithLogitsLoss(pos_weight=trainset.labels_weights.to(DEVICE))
    loss_weights = [1.0] + misc.loss_weights
    task_prob = [1 - misc.mt_task_prob, misc.mt_task_prob / 2., misc.mt_task_prob / 2.]

    if model_type in ['singletask', 'multitask', 'dualnet'] and len(learning_rate) > 1:
        # each branch has custom learning rate
        optim_params = [{'params': model.frontal_model.parameters(), 'lr': learning_rate[0]},
                        {'params': model.lateral_model.parameters(), 'lr': learning_rate[1]},
                        {'params': model.classifier.parameters(), 'lr': learning_rate[2]}]
    else:
        # one lr for all
        optim_params = [{'params': model.parameters(), 'lr': learning_rate[0]}]

    if misc.learn_loss_coeffs:
        temperature = torch.ones(size=(3,), requires_grad=True, device=DEVICE).float()
        temperature_lr = learning_rate[-1] if len(learning_rate) > 3 else learning_rate[0]
        optim_params.append({'params': temperature, 'lr': temperature_lr})

    # Optimizer
    optimizer, scheduler = create_opt_and_sched(optim=optim, params=optim_params, lr=learning_rate[0], other_args=misc)
    start_epoch = 1

    # Resume training if possible
    latest_ckpt_file = join(output_dir, f'{target}-latest.tar')
    if isfile(latest_ckpt_file):
        with torch.load(latest_ckpt_file) as checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        evaluator.load_saved()
        start_epoch = int(evaluator.eval_df.epoch.iloc[-1])
        logger.info(f"Resumed at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, nb_epoch + 1):
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
                loss = losses_dict[select]
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
                logger.info(f'[{epoch}, {i + 1:5}] loss: {running_loss:.5f}')
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
        torch.save(model.state_dict(), join(output_dir, f'{target}-e{epoch}.pt'))

        # Remove all batches weights
        weights_files = glob(join(output_dir, f'{target}-e{epoch}-i*.pt'))
        for file in weights_files:
            os.remove(file)

    logger.info(evaluator.eval_df.auc.iloc[-1])
    return - evaluator.eval_df.auc.iloc[-1]


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
    parser.add_argument('--learning_rate', type=str, default=[0.0001])
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
                        help='For joint. Drop either view with p/2 and keep both views with 1-p')
    parser.add_argument('--mt-task-prob', type=float, default=0.0,
                        help='Curriculum learning probs. Drop either task with p/2 and keep both views with 1-p')
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

    args = parser.parse_args()

    np.set_printoptions(suppress=True, precision=4)

    if args.exp_name:
        args.output_dir = args.output_dir + "-" + args.exp_name

    if args.dropout >= 1:
        args.dropout /= 10.

    if args.data_dir == "CLUSTER":
        args.data_dir = os.environ.get('DATADIR')

    if args.target != 'joint':
        mlflow_name = args.target
        exp_name = args.arch
    else:
        mlflow_name = args.model_type
        exp_name = args.model_type

    mlflow.set_experiment(f'lateral-view-{mlflow_name}')
    mlflow.start_run(run_name=f'{exp_name}-run{args.exp_name}')

    logging.basicConfig(level=logging.INFO)

    # will log to a file if provided
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    args.learning_rate = [float(lr) for lr in eval(args.learning_rate)]

    logger.info(args)
    val_loss = train(args.data_dir, args.csv_path, args.splits_path, args.output_dir, target=args.target,
                     nb_epoch=args.epochs, learning_rate=args.learning_rate, batch_size=args.batch_size,
                     dropout=args.dropout, optim=args.optim, min_patients_per_label=args.min_patients, seed=args.seed,
                     model_type=args.model_type, architecture=args.arch, data_augmentation=args.data_augmentation,
                     misc=args)

    report_results([dict(name='val_auc', type='objective', value=val_loss)])
