import argparse
from os.path import join, exists

import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, split_dataset
from densenet import DenseNet, add_dropout
from hemis import Hemis, add_dropout_hemis

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import pandas as pd
import pickle

torch.manual_seed(42)
np.random.seed(42)


def test(data_dir, csv_path, splits_path, output_dir, #weights_file, 
         target='pa', batch_size=1, dropout=True, pretrained=False, 
         min_patients_per_label=100, seed=666):
    assert target in ['pa', 'l', 'joint']
    output_dir = output_dir.format(seed)

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
    if pretrained:
        in_channels = 3
    else:
        in_channels = 2 if target == 'joint' else 1
    
    if target == 'joint':
        model = Hemis(num_classes=testset.nb_labels, in_channels=1)
    else:
        model = DenseNet(num_classes=testset.nb_labels, in_channels=in_channels)

    # Add dropout
    if dropout:
        model = add_dropout(model, p=0.2) if target != 'joint' else add_dropout_hemis(model, p=0.2)

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
    for data in testloader:
        if target == 'pa':
            input, label = data['PA'].to(device), data['encoded_labels'].to(device)
        elif target == 'l':
            input, label = data['L'].to(device), data['encoded_labels'].to(device)
        else:
            pa, l, label = data['PA'].to(device), data['L'].to(device), data['encoded_labels'].to(device)
            input = torch.cat([pa, l], dim=1)

        # Forward
        output = model(input)
        output = torch.sigmoid(output)

        # Save predictions
        y_preds.append(output.data.cpu().numpy())
        y_true.append(label.data.cpu().numpy())

    y_preds = np.vstack(y_preds)
    y_true = np.vstack(y_true)
    
    np.save("{}_preds_{}".format(target,seed), y_preds)
    np.save("{}_true_{}".format(target,seed), y_true)

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
        pickle.dump({'auc':auc, 'prc':prc}, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')
    parser.add_argument('data_dir', type=str)
    parser.add_argument('csv_path', type=str)
    parser.add_argument('splits_path', type=str)
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--pretrained', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--min_patients', type=int, default=50)
    args = parser.parse_args()
    print(args)
    test(args.data_dir, args.csv_path, args.splits_path, args.output_dir, 
         target=args.target, 
         batch_size=args.batch_size, pretrained=args.pretrained, 
         min_patients_per_label=args.min_patients,  seed=args.seed)
