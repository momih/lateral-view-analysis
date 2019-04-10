from os.path import join
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset import PCXRayDataset


def _load_metrics(results_dir, target):
    return pd.read_csv(join(results_dir, '{}-metrics.csv'.format(target)),
                       usecols=['accuracy', 'auc', 'prc', 'loss', 'epoch', 'error'], low_memory=False)


def _load_per_label(results_dir, target, metric):
    with open(join(results_dir, '{}-val_{}.pkl'.format(target, metric)), 'rb') as f:
        res = pickle.load(f)

    return res


def plot_metrics(results_dir):
    pa_metrics = _load_metrics(results_dir, 'pa')
    l_metrics = _load_metrics(results_dir, 'l')
    joint_metrics = _load_metrics(results_dir, 'joint')

    sns.set(style="whitegrid")

    x = np.arange(1, 101)
    plt.plot(x, pa_metrics['accuracy'], label='PA')
    plt.plot(x, l_metrics['accuracy'], label='L')
    plt.plot(x, joint_metrics['accuracy'], label='PA+L')
    plt.title('Average accuracy')
    plt.legend()
    plt.ylim((0., 1.))
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_accuracy.png'))
    plt.close()

    x = np.arange(1, 101)
    plt.plot(x, pa_metrics['auc'], label='PA')
    plt.plot(x, l_metrics['auc'], label='L')
    plt.plot(x, joint_metrics['auc'], label='PA+L')
    plt.title('Weighted auc')
    plt.legend()
    plt.ylim((0., 1.))
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_auc.png'))
    plt.close()

    x = np.arange(1, 101)
    plt.plot(x, pa_metrics['prc'], label='PA')
    plt.plot(x, l_metrics['prc'], label='L')
    plt.plot(x, joint_metrics['prc'], label='PA+L')
    plt.title('Weighted precision')
    plt.legend()
    plt.ylim((0., 1.))
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_precision.png'))
    plt.close()


def plot_per_label(results_dir, labels_list):
    epoch = 60

    pa_prc = _load_per_label(results_dir, 'pa', 'prc')[epoch]
    l_prc = _load_per_label(results_dir, 'l', 'prc')[epoch]
    joint_prc = _load_per_label(results_dir, 'joint', 'prc')[epoch]

    d = {'PA': pa_prc, 'L': l_prc, 'PA+L': joint_prc, 'Labels': labels_list}
    df = pd.DataFrame(d)
    df = df.sort_values('L', ascending=False)
    df = pd.melt(df, id_vars="Labels", var_name="View", value_name="Precision")
    # df = df.sort_values('Precision', ascending=False)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 13))

    g = sns.barplot(x='Precision', y='Labels', hue='View', data=df)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_precision_per_label.png'))
    plt.close()

    ######################
    pa_auc = _load_per_label(results_dir, 'pa', 'auc')[epoch]
    l_auc = _load_per_label(results_dir, 'l', 'auc')[epoch]
    joint_auc = _load_per_label(results_dir, 'joint', 'auc')[epoch]

    d = {'PA': pa_auc, 'L': l_auc, 'PA+L': joint_auc, 'Labels': labels_list}
    df = pd.DataFrame(d)
    df = df.sort_values('L', ascending=False)
    df = pd.melt(df, id_vars="Labels", var_name="View", value_name="Area under Curve")
    # df = df.sort_values('Precision', ascending=False)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 13))

    g = sns.barplot(x='Area under Curve', y='Labels', hue='View', data=df)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_auc_per_label.png'))
    plt.close()


def plot_per_label_diff(results_dir, labels_list):
    pa_prc = _load_per_label(results_dir, 'pa', 'prc')[-1]
    l_prc = _load_per_label(results_dir, 'l', 'prc')[-1]
    joint_prc = _load_per_label(results_dir, 'joint', 'prc')[-1]

    d = {'PA': pa_prc, 'L': l_prc, 'PA+L': joint_prc, 'Labels': labels_list}
    df = pd.DataFrame(d)
    df = df.sort_values('L', ascending=False)
    df = pd.melt(df, id_vars="Labels", var_name="View", value_name="Precision")
    # df = df.sort_values('Precision', ascending=False)

    sns.set(style="whitegrid")

    fig, ax = plt.subplots(figsize=(10, 13))

    g = sns.barplot(x='Precision', y='Labels', hue='View', data=df)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_precision_per_label_diff.png'))
    plt.close()


if __name__ == "__main__":
    num_patients = 200
    results_dir = './models/p{}'.format(num_patients)
    cohort_file = './data/joint_PA_L.csv'
    img_dir = './data/processed'

    dataset = PCXRayDataset(img_dir, cohort_file, None, min_patients_per_label=num_patients)

    # plot_metrics(results_dir)
    plot_per_label(results_dir, dataset.labels)
    # plot_per_label_diff(results_dir, dataset.labels)
