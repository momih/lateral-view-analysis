import argparse
from os.path import join, exists, isfile

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from dataset import PCXRayDataset, Normalize, ToTensor, split_dataset
from evaluate import get_model_preds
from models import create_model

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_metrics(y_true, y_preds, extra=''):
    print(y_preds.shape)
    per_label_auc = roc_auc_score(y_true, y_preds, average=None)
    per_label_prc = average_precision_score(y_true, y_preds, average=None)
    print(f"AUC {extra}: \n{per_label_auc}\n\nPRC {extra}:\n{per_label_prc}\n")
    metrics = {'accuracy': accuracy_score(y_true, np.where(y_preds > 0.5, 1, 0)),
               'auc': roc_auc_score(y_true, y_preds),
               'prc': average_precision_score(y_true, y_preds),
               'auc_weighted': roc_auc_score(y_true, y_preds, average='weighted'),
               'prc_weighted': average_precision_score(y_true, y_preds, average='weighted')}

    return metrics, per_label_auc, per_label_prc


def test(data_dir, csv_path, splits_path, output_dir, target='pa',
         batch_size=1, pretrained=False, min_patients_per_label=100, seed=666,
         model_type='hemis', architecture='densenet121', misc=None):
    assert target in ['pa', 'l', 'joint']

    print(f"\n\nTesting seed {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    extra = misc.extra
    name = output_dir.split("/")[-1].format('') + extra
    output_dir = output_dir.format(seed)
    splits_path = splits_path.format(seed)

    # Save predictions
    predsdir = join(output_dir, '..', 'test_outs')
    resultsfile = join(output_dir, '..', 'auc-test.csv')

    if not isfile(resultsfile):
        columns = ['expt', 'seed', 'accuracy', 'auc', 'auc_weighted', 'prc', 'prc_weighted']
        test_metrics_df = pd.DataFrame(columns=columns)
    else:
        test_metrics_df = pd.read_csv(resultsfile)



    if not exists(splits_path):
        split_dataset(csv_path, splits_path)

    print("Test mode: {}".format(target))
    print('Device that will be used is: {0}'.format(DEVICE))

    # Load data
    preprocessing = Compose([Normalize(), ToTensor()])

    testset = PCXRayDataset(data_dir, csv_path, splits_path, transform=preprocessing, dataset='test',
                            pretrained=pretrained, min_patients_per_label=min_patients_per_label)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("{0} patients in test set.".format(len(testset)))

    # Find best weights
    metricsdf = pd.read_csv(join(output_dir, f'{target}-metrics.csv'))
    best_epoch = int(metricsdf.idxmax()['auc'])
    weights_file = join(output_dir, '{}-e{}.pt'.format(target, best_epoch))

    # Create model and load best weights
    model = create_model(model_type, num_classes=testset.nb_labels, target=target,
                         architecture=architecture, dropout=0.0, otherargs=misc)
    try:
        model.load_state_dict(torch.load(weights_file))
    except:
        # Issue in loading weights if trained on multiple GPUs
        state_dict = torch.load(weights_file, map_location='cpu')
        for key in list(state_dict.keys()):
            if 'conv' in key or 'classifier' in key:
                if '.0.' in key:
                    new_key = key.replace(".0.", '.')
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]

        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    if misc.test_multi:
        y_true, y_preds, _ = get_model_preds(model, dataloader=testloader, target=target, model_type=model_type,
                                             vote_at_test=misc.vote_at_test, progress_bar=True)
        metrics, per_label_auc, per_label_prc = get_metrics(y_true, y_preds)
        row = {'expt': name, 'seed': seed, **metrics}
        print(row)

        test_metrics_df = test_metrics_df.append(row, ignore_index=True)
        test_metrics_df.to_csv(resultsfile, index=False)

        savepreds = {'y_true': y_true, 'y_preds': y_preds, 'meta': row}
        saveauc = {'auc': per_label_auc, 'prc': per_label_prc, 'meta': row}

    else:
        print('Loading save files')
        _arr = np.load(join(predsdir, f'preds-{name}{extra}_{seed}-{target}'), allow_pickle=True)
        savepreds = {k: _arr[k] for k in _arr.keys()}

        _arr = np.load(join(predsdir, f'auc-{name}{extra}_{seed}-{target}'), allow_pickle=True)
        saveauc = {k: _arr[k] for k in _arr.keys()}

    for view in misc.test_single:
        print(f"Testing on only {view}")
        model.test_only_one = 0
        y_true_pa_only, y_preds_pa_only, _ = get_model_preds(model, dataloader=testloader, target=target,
                                                             test_on=view, model_type=model_type,
                                                             vote_at_test=misc.vote_at_test, progress_bar=True)

        metrics, per_label_auc, per_label_prc = get_metrics(y_true_pa_only, y_preds_pa_only)
        row = {'expt': name + f'{view}_only', 'seed': seed, **metrics}
        print(row)

        savepreds[f'y_true_{view}_only'] = y_true_pa_only
        savepreds[f'y_preds_{view}_only'] = y_preds_pa_only
        savepreds[f'meta_{view}_only'] = row

        test_metrics_df = test_metrics_df.append(row, ignore_index=True)
        test_metrics_df.to_csv(resultsfile, index=False)

        saveauc[f'auc_{view}_only'] = per_label_auc
        saveauc[f'prc_{view}_only'] = per_label_prc
        saveauc[f'meta_{view}_only'] = row


    np.savez(join(predsdir, f'preds-{name}{extra}_{seed}-{target}'), **savepreds)
    np.savez(join(predsdir, f'auc-{name}{extra}_{seed}-{target}'), **saveauc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Usage')
    # Paths
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--splits_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--extra', type=str, default='')

    # Model params
    parser.add_argument('--arch', type=str, default='densenet121')
    parser.add_argument('--model-type', type=str, default='hemis')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--vote-at-test', action='store_true')
    parser.add_argument('--test-multi', action='store_true')
    parser.add_argument('--test-single', default=['pa', 'l'], nargs='*')

    # Hyperparams
    parser.add_argument('--batch_size', type=int, default=1)

    # Dataset params
    parser.add_argument('--target', type=str, default='pa')
    parser.add_argument('--min_patients', type=int, default=50)
    parser.add_argument('--start_seed', type=int, default=1)
    parser.add_argument('--n_seeds', type=int, default=5)

    # Other optional arguments
    parser.add_argument('--merge', type=int, default=3)
    parser.add_argument('--mt-combine-at', dest='combine', type=str, default='prepool')
    parser.add_argument('--mt-join', dest='join', type=str, default='concat')
    parser.add_argument('--drop-view-prob', type=float, default=0.0,
                        help="""For Hemis, HemisConcat and Multitask. 
                        Drop either view with prob/2 and keep both views with 1-prob""")

    args = parser.parse_args()
    if args.exp_name:
        args.output_dir = args.output_dir + "-" + args.exp_name
    print(args)
    for _seed in range(args.start_seed, args.n_seeds + 1):
        test(args.data_dir, args.csv_path, args.splits_path, args.output_dir, target=args.target,
             batch_size=args.batch_size, pretrained=args.pretrained,
             min_patients_per_label=args.min_patients, seed=_seed,
             model_type=args.model_type, architecture=args.arch, misc=args)
