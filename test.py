import argparse
import pickle
from os.path import join, exists

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from tqdm import tqdm

from dataset import PCXRayDataset, Normalize, ToTensor, split_dataset
from models import DenseNet, HeMIS, HeMISConcat, FrontalLateralMultiTask, ResNet
from models import get_densenet_params, get_resnet_params


def test(data_dir, csv_path, splits_path, output_dir, logdir='./logs', target='pa',
         batch_size=1, pretrained=False, min_patients_per_label=100, seed=666,
         model_type='hemis', architecture='densenet121', other_args=None):
    assert target in ['pa', 'l', 'joint']

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir.format(seed)
    output_dir = join(logdir, output_dir)

    splits_path = splits_path.format(seed)

    print("Test mode: {}".format(target))

    if not exists(splits_path):
        split_dataset(csv_path, splits_path)

    # Find device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device that will be used is: {0}'.format(device))

    # Load data
    preprocessing = Compose([Normalize(), ToTensor()])

    testset = PCXRayDataset(data_dir, csv_path, splits_path, transform=preprocessing, dataset='test',
                            pretrained=pretrained, min_patients_per_label=min_patients_per_label)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("{0} patients in test set.".format(len(testset)))

    # Load model
    in_channels = 3 if pretrained else 1

    if target == 'joint':
        if model_type == 'concat':
            model = HeMISConcat(num_classes=testset.nb_labels, in_channels=1)
        elif model_type == 'multitask':
            model = FrontalLateralMultiTask(num_classes=testset.nb_labels, combine_at=other_args.combine,
                                            join_how=other_args.join, architecture=architecture)
        else:
            model = HeMIS(num_classes=testset.nb_labels, in_channels=1, merge_at=other_args.merge)
    else:
        if 'resnet' in architecture:
            modelparams = get_resnet_params(architecture)
            model = ResNet(num_classes=testset.nb_labels, in_channels=in_channels, **modelparams)
            model_type = 'resnet'
        else:
            # Use DenseNet by default
            modelparams = get_densenet_params(architecture)
            model = DenseNet(num_classes=testset.nb_labels, in_channels=in_channels, **modelparams)
            model_type = 'densenet'

    # Find best weights
    df_file = '{}-metrics.csv'.format(target)
    metricsdf = pd.read_csv(join(output_dir, df_file))
    best_epoch = int(metricsdf.idxmax()['auc'])

    # Load trained weights
    weights_file = join(output_dir, '{}-e{}.pt'.format(target, best_epoch))
    model.load_state_dict(torch.load(weights_file))

    model.to(device)
    model.eval()

    y_preds = []
    y_true = []
    for data in tqdm(testloader):
        if target == 'pa':
            input, label = data['PA'].to(device), data['encoded_labels'].to(device)
        elif target == 'l':
            input, label = data['L'].to(device), data['encoded_labels'].to(device)
        else:
            pa, l, label = data['PA'].to(device), data['L'].to(device), data['encoded_labels'].to(device)
            input = [pa, l]

        # Forward
        output = model(input)
        if model_type == 'multitask':
            if other_args.vote_at_test:
                output = torch.stack(output, dim=1).mean(dim=1)
            else:
                output = output[0]

        output = torch.sigmoid(output)

        # Save predictions
        y_preds.append(output.data.cpu().numpy())
        y_true.append(label.data.cpu().numpy())

    y_preds = np.vstack(y_preds)
    y_true = np.vstack(y_true)

    np.save(join(output_dir, "{}_preds_{}".format(target, seed)), y_preds)
    np.save(join(output_dir, "{}_true_{}".format(target, seed)), y_true)

    print(y_preds.shape)
    auc = roc_auc_score(y_true, y_preds, average=None)
    print("auc")
    print(auc)
    print()
    prc = average_precision_score(y_true, y_preds, average=None)
    print("prc")
    print(prc)
    print()
    metrics = {'accuracy': accuracy_score(y_true, np.where(y_preds > 0.5, 1, 0)),
               'auc': roc_auc_score(y_true, y_preds, average='weighted'),
               'prc': average_precision_score(y_true, y_preds, average='weighted')}

    print(metrics)
    with open(join(output_dir, '{}-seed{}-test.pkl'.format(target, seed)), 'wb') as f:
        pickle.dump({'auc': auc, 'prc': prc}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')
    # Paths
    parser.add_argument('data_dir', type=str)
    parser.add_argument('csv_path', type=str)
    parser.add_argument('splits_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--logdir', type=str, default='./logs')

    # Model params
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--model-type', type=str, default='hemis')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--vote-at-test', action='store_true')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)

    # Dataset params
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--min_patients', type=int, default=50)
    parser.add_argument('--seed', type=int, default=666)

    # Other optional arguments
    parser.add_argument('--merge', type=int, default=2)
    parser.add_argument('--mt-combine-at', dest='combine', type=str, default='prepool')
    parser.add_argument('--mt-join', dest='join', type=str, default='concat')
    args = parser.parse_args()
    print(args)

    test(args.data_dir, args.csv_path, args.splits_path, args.output_dir, target=args.target,
         logdir=args.logdir, batch_size=args.batch_size, pretrained=args.pretrained,
         min_patients_per_label=args.min_patients, seed=args.seed,
         model_type=args.model_type, architecture=args.arch, other_args=args)
