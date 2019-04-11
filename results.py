from os.path import join
import pickle

import matplotlib.patches as mpatches
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
    plt.xlim([0.5, 1.])
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_auc_per_label.png'))
    plt.close()


def plot_per_label_diff(results_dir, labels_list):
    epoch = 60

    diff_map = {0: 'PA', 1: 'L', 2: 'PA+L', 3: 'Indifferent'}
    clr_map = {'PA': 'b', 'L': 'r', 'PA+L': 'g', 'Indifferent': 'm'}

    pa_auc = _load_per_label(results_dir, 'pa', 'auc')[epoch]
    l_auc = _load_per_label(results_dir, 'l', 'auc')[epoch]
    joint_auc = _load_per_label(results_dir, 'joint', 'auc')[epoch]
    arr_auc = np.array([pa_auc, l_auc, joint_auc])
    diff_auc = np.max(arr_auc, axis=0) - np.min(arr_auc, axis=0)
    amax_auc = np.argmax(arr_auc, axis=0)
    amax_auc[np.where(diff_auc < 0.02)] = 3
    amax_auc = [diff_map[d] for d in list(amax_auc)]

    pa_prc = _load_per_label(results_dir, 'pa', 'prc')[epoch]
    l_prc = _load_per_label(results_dir, 'l', 'prc')[epoch]
    joint_prc = _load_per_label(results_dir, 'joint', 'prc')[epoch]
    arr_prc = np.array([pa_prc, l_prc, joint_prc])
    diff_prc = np.max(arr_prc, axis=0) - np.min(arr_prc, axis=0)
    amax_prc = np.argmax(arr_prc, axis=0)
    amax_prc[np.where(diff_prc < 0.01)] = 3
    amax_prc = [diff_map[d] for d in list(amax_prc)]

    d = {
        'Labels': labels_list,
        'Best view (AuC)': amax_auc,
        'Area under curve': np.max(arr_auc, axis=0),
        'Diff auc': diff_auc,
        'Min auc': np.min(arr_auc, axis=0),
        'Best view (Precision)': amax_prc,
        'Precision': np.max(arr_prc, axis=0),
        'Diff Prc': diff_prc,
        'Min prc': np.min(arr_prc, axis=0)
    }
    df = pd.DataFrame(d)
    df = df.sort_values(['Best view (AuC)', 'Area under curve'], ascending=False)

    sns.set(style="whitegrid")

    clrs_auc = [clr_map[x] for x in df['Best view (AuC)']]

    pal_patch = mpatches.Patch(color='g', label='PA+L')
    pa_patch = mpatches.Patch(color='b', label='PA')
    l_patch = mpatches.Patch(color='r', label='L')
    ind_patch = mpatches.Patch(color='m', label='Indifferent')

    fig, ax = plt.subplots(figsize=(10, 13))

    sns.set_color_codes("pastel")
    sns.barplot(x='Area under curve', y='Labels', data=df, palette=clrs_auc)
    sns.set_color_codes("deep")
    sns.barplot(x='Min auc', y='Labels', data=df, palette=clrs_auc)

    ax.legend(handles=[pal_patch, pa_patch, l_patch, ind_patch], ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="", xlabel="")
    ax.set_title('Best AUC for each label')
    plt.xlim((0.5, 1.))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_auc_per_label_diff.png'))
    plt.close()

    ############
    df = df.sort_values(['Best view (Precision)', 'Precision'], ascending=False)

    clrs_prc = [clr_map[x] for x in df['Best view (Precision)']]

    fig, ax = plt.subplots(figsize=(10, 13))

    sns.set_color_codes("pastel")
    sns.barplot(x='Precision', y='Labels', data=df, palette=clrs_prc)
    sns.set_color_codes("deep")
    sns.barplot(x='Min prc', y='Labels', data=df, palette=clrs_prc)

    ax.legend(handles=[pal_patch, pa_patch, l_patch, ind_patch], ncol=2, loc="lower right", frameon=True)
    ax.set(ylabel="", xlabel="")
    ax.set_title('Best precision for each label')
    plt.xlim((0., 0.7))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir, '_results_precision_per_label_diff.png'))
    plt.close()


if __name__ == "__main__":
    num_patients = 200
    results_dir = './models/p{}lr'.format(num_patients)
    cohort_file = './data/joint_PA_L.csv'
    img_dir = './data/processed'

    dataset = PCXRayDataset(img_dir, cohort_file, None, min_patients_per_label=num_patients)
    labels_list = ['{} ({})'.format(l, c // 2) for l, c in zip(dataset.labels, dataset.labels_count)]

    plot_metrics(results_dir)
    plot_per_label(results_dir, labels_list)
    plot_per_label_diff(results_dir, labels_list)
