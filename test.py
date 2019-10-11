import argparse
import pickle
from os.path import join, exists

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, split_dataset
from models import create_model
from evaluate import get_model_preds

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test(data_dir, csv_path, splits_path, output_dir, target='pa',
         batch_size=1, pretrained=False, min_patients_per_label=100, seed=666,
         model_type='hemis', architecture='densenet121', misc=None):
    assert target in ['pa', 'l', 'joint']

    torch.manual_seed(seed)
    np.random.seed(seed)

    output_dir = output_dir.format(seed)

    splits_path = splits_path.format(seed)

    print("Test mode: {}".format(target))

    if not exists(splits_path):
        split_dataset(csv_path, splits_path)

    # Find device
    print('Device that will be used is: {0}'.format(DEVICE))

    # Load data
    preprocessing = Compose([Normalize(), ToTensor()])

    testset = PCXRayDataset(data_dir, csv_path, splits_path, transform=preprocessing, dataset='test',
                            pretrained=pretrained, min_patients_per_label=min_patients_per_label)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)

    print("{0} patients in test set.".format(len(testset)))
    # Find best weights
    metricsdf = pd.read_csv(join(output_dir, f'{target}-metrics.csv'))
    best_epoch = int(metricsdf.idxmax()['auc'])

    # Create model and load best weights
    model = create_model(model_type, num_classes=testset.nb_labels, target=target,
                         architecture=architecture, dropout=0.0, otherargs=misc)
    model.to(DEVICE)
    weights_file = join(output_dir, '{}-e{}.pt'.format(target, best_epoch))
    model.load_state_dict(torch.load(weights_file))
    model.eval()

    y_true, y_preds, _ = get_model_preds(model, dataloader=testloader, target=target,
                                         model_type=model_type, vote_at_test=misc.vote_at_test, progress_bar=True)

    np.save(join(output_dir, "{}_preds_{}".format(target, seed)), y_preds)
    np.save(join(output_dir, "{}_true_{}".format(target, seed)), y_true)

    print(y_preds.shape)
    auc = roc_auc_score(y_true, y_preds, average=None)
    prc = average_precision_score(y_true, y_preds, average=None)

    print("auc")
    print(auc)
    print()
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
    parser.add_argument('--drop-view-prob', type=float, default=0.0,
                        help='For Hemis, HemisConcat and Multitask. Drop either view with prob/2 and keep both views with 1-prob')

    args = parser.parse_args()
    print(args)

    test(args.data_dir, args.csv_path, args.splits_path, args.output_dir, target=args.target,
         batch_size=args.batch_size, pretrained=args.pretrained,
         min_patients_per_label=args.min_patients, seed=args.seed,
         model_type=args.model_type, architecture=args.arch, misc=args)
