from os.path import join

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from dataset import PCXRayDataset


def plot_per_label_diff(results_dir, labels_list, seed_list, nb_patients):
    diff_map = {0: 'PA', 1: 'L', 2: 'PA+L', 3: 'Indifferent'}
    clr_map = {'PA': 'b', 'L': 'r', 'PA+L': 'g', 'Indifferent': 'm'}

    def _load_test(results_dir, seed, target):
        # TODO test file loading
        arr_file = join(results_dir, f'{target}-seed{seed}-testauc.npz')
        test_auc = np.load(arr_file, allow_pickle=True)['auc']
        return test_auc

    pa_list, l_list, joint_list = [], [], []

    for seed in seed_list:
        pa_list.append(_load_test(results_dir, seed, 'pa'))
        l_list.append(_load_test(results_dir, seed, 'l'))
        joint_list.append(_load_test(results_dir, seed, 'joint'))

    pa_list = np.array(pa_list)
    l_list = np.array(l_list)
    joint_list = np.array(joint_list)
    d = {'Labels': labels_list,
         'pa_mean': pa_list.mean(0), 'pa_std': pa_list.std(0),
         'l_mean': l_list.mean(0), 'l_std': l_list.std(0),
         'joint_mean': joint_list.mean(0), 'joint_std': joint_list.std(0)}

    arr_auc = np.vstack([d['pa_mean'], d['l_mean'], d['joint_mean']])
    arr_std = np.vstack([d['pa_std'], d['l_std'], d['joint_std']])
    amax_auc = np.argmax(arr_auc, axis=0)
    diff_auc = np.abs(arr_auc[0] - np.max(arr_auc[1:], axis=0))
    # diff_auc = np.abs(arr_auc[0] - arr_auc[1])

    # arr_auc = np.array([pa_auc, l_auc, joint_auc])
    # diff_auc = np.max(arr_auc, axis=0) - np.min(arr_auc, axis=0)
    # amax_auc = np.argmax(arr_auc, axis=0)
    # amax_auc[np.where(diff_auc < 0.02)] = 3
    # amax_auc = [diff_map[d] for d in list(amax_auc)]

    diff_auc = np.abs(arr_auc.max(axis=0) - arr_auc)
    diff_auc = np.min(np.where(diff_auc == 0, diff_auc.max(), diff_auc), axis=0)

    d['Area under curve'] = np.max(arr_auc, axis=0)
    d['std'] = arr_std[amax_auc][0]
    d['diff'] = np.max(arr_auc, axis=0) - diff_auc

    amax_auc[np.where(diff_auc < d['std'])] = 3

    d['Best view'] = [diff_map[d] for d in list(amax_auc)]
    d['clrs'] = [clr_map[x] for x in d['Best view']]

    df = pd.DataFrame(d)
    df = df.sort_values(['Best view', 'Area under curve'], ascending=False)

    sns.set(style="whitegrid")

    pal_patch = mpatches.Patch(color='g', label='PA+L')
    pa_patch = mpatches.Patch(color='b', label='PA')
    l_patch = mpatches.Patch(color='r', label='L')
    ind_patch = mpatches.Patch(color='m', label='Indifferent')

    fig = plt.figure(figsize=(36, 12), dpi=220)
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    condition = df['Best view'] == 'PA'

    sns.set_color_codes("pastel")
    sns.barplot(x='Area under curve', y='Labels', data=df[condition], palette=df[condition]['clrs'], ax=ax1)
    sns.barplot(x='Area under curve', y='Labels', data=df[~condition], palette=df[~condition]['clrs'], ax=ax2)
    sns.set_color_codes("deep")
    sns.barplot(x='diff', y='Labels', data=df[condition], palette=df[condition]['clrs'], ax=ax1)
    sns.barplot(x='diff', y='Labels', data=df[~condition], palette=df[~condition]['clrs'], ax=ax2)

    ax1.errorbar(x=df[condition]['Area under curve'], y=np.arange(len(df[condition])),
                 xerr=df[condition]['std'], fmt='none', c='k', capsize=3)
    ax2.errorbar(x=df[~condition]['Area under curve'], y=np.arange(len(df[~condition])),
                 xerr=df[~condition]['std'], fmt='none', c='k', capsize=3)

    ax2.legend(handles=[pa_patch, l_patch, ind_patch], ncol=2, loc="lower right", frameon=True, fontsize=24)
    ax1.set(ylabel="", xlabel="")
    ax2.set(ylabel="", xlabel="")
    fig.suptitle('Best AUC for each label ({} patients)'.format(nb_patients), fontsize=28)
    ax1.set_xlim((0.5, 1.))
    ax2.set_xlim((0.5, 1.))
    ax1.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    # ax.set_ylim((-1, len(labels_list)))
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    plt.savefig(join(results_dir[:-3], '_results_auc_per_label_diff.png'))
    plt.close()


if __name__ == "__main__":
    num_patients = 100
    results_dir = './models/s{}'
    cohort_file = './data/joint_PA_L.csv'
    img_dir = './data/processed'
    seed_list = [0, 1, 2, 3]

    dataset = PCXRayDataset(img_dir, cohort_file, None, min_patients_per_label=num_patients)
    labels_list = ['{} ({})'.format(l.replace('normal', 'no finding'), c // 2) for l, c in
                   zip(dataset.labels, dataset.labels_count)]

    # plot_metrics(results_dir)
    # plot_per_label(results_dir, labels_list)
    plot_per_label_diff(results_dir, labels_list, seed_list, len(dataset))

#
# def plot_metrics(results_dir):
#     epoch = 40
#
#     def _load_metrics(results_dir, target):
#         return pd.read_csv(join(results_dir, '{}-metrics.csv'.format(target)),
#                            usecols=['accuracy', 'auc', 'prc', 'loss', 'epoch', 'error'], low_memory=False)
#
#     pa_metrics = _load_metrics(results_dir, 'pa')
#     l_metrics = _load_metrics(results_dir, 'l')
#     joint_metrics = _load_metrics(results_dir, 'joint')
#
#     sns.set(style="whitegrid")
#
#     x = np.arange(1, epoch + 1)
#     plt.plot(x, pa_metrics['accuracy'], label='PA')
#     plt.plot(x, l_metrics['accuracy'], label='L')
#     plt.plot(x, joint_metrics['accuracy'], label='PA+L')
#     plt.title('Average accuracy')
#     plt.legend()
#     plt.ylim((0., 1.))
#     plt.tight_layout()
#     plt.savefig(join(results_dir, '_results_accuracy.png'))
#     plt.close()
#
#     x = np.arange(1, epoch + 1)
#     plt.plot(x, pa_metrics['auc'], label='PA')
#     plt.plot(x, l_metrics['auc'], label='L')
#     plt.plot(x, joint_metrics['auc'], label='PA+L')
#     plt.title('Weighted auc')
#     plt.legend()
#     plt.ylim((0., 1.))
#     plt.tight_layout()
#     plt.savefig(join(results_dir, '_results_auc.png'))
#     plt.close()
#
#     x = np.arange(1, epoch + 1)
#     plt.plot(x, pa_metrics['prc'], label='PA')
#     plt.plot(x, l_metrics['prc'], label='L')
#     plt.plot(x, joint_metrics['prc'], label='PA+L')
#     plt.title('Weighted precision')
#     plt.legend()
#     plt.ylim((0., 1.))
#     plt.tight_layout()
#     plt.savefig(join(results_dir, '_results_precision.png'))
#     plt.close()
#
#
# def plot_per_label(results_dir, labels_list):
#     epoch = 40
#
#     def _load_per_label(results_dir, target, metric):
#         with open(join(results_dir, '{}-val_{}.pkl'.format(target, metric)), 'rb') as f:
#             res = pickle.load(f)
#         return res
#
#     pa_prc = _load_per_label(results_dir, 'pa', 'prc')[epoch]
#     l_prc = _load_per_label(results_dir, 'l', 'prc')[epoch]
#     joint_prc = _load_per_label(results_dir, 'joint', 'prc')[epoch]
#
#     d = {'PA': pa_prc, 'L': l_prc, 'PA+L': joint_prc, 'Labels': labels_list}
#     df = pd.DataFrame(d)
#     df = df.sort_values('L', ascending=False)
#     df = pd.melt(df, id_vars="Labels", var_name="View", value_name="Precision")
#     # df = df.sort_values('Precision', ascending=False)
#
#     sns.set(style="whitegrid")
#
#     fig, ax = plt.subplots(figsize=(10, 13))
#
#     g = sns.barplot(x='Precision', y='Labels', hue='View', data=df)
#     sns.despine(left=True, bottom=True)
#     plt.tight_layout()
#     plt.savefig(join(results_dir, '_results_precision_per_label.png'))
#     plt.close()
#
#     ######################
#     pa_auc = _load_per_label(results_dir, 'pa', 'auc')[epoch]
#     l_auc = _load_per_label(results_dir, 'l', 'auc')[epoch]
#     joint_auc = _load_per_label(results_dir, 'joint', 'auc')[epoch]
#
#     d = {'PA': pa_auc, 'L': l_auc, 'PA+L': joint_auc, 'Labels': labels_list}
#     df = pd.DataFrame(d)
#     df = df.sort_values('L', ascending=False)
#     df = pd.melt(df, id_vars="Labels", var_name="View", value_name="Area under Curve")
#     # df = df.sort_values('Precision', ascending=False)
#
#     sns.set(style="whitegrid")
#
#     fig, ax = plt.subplots(figsize=(10, 13))
#
#     g = sns.barplot(x='Area under Curve', y='Labels', hue='View', data=df)
#     sns.despine(left=True, bottom=True)
#     plt.xlim([0.5, 1.])
#     plt.tight_layout()
#     plt.savefig(join(results_dir, '_results_auc_per_label.png'))
#     plt.close()
