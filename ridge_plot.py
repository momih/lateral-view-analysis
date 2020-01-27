import argparse
import csv
import datetime
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

logger = logging.getLogger(__name__)


def parse_args(args):
    """
    Argument parser for this script.

    Args:
        args (list of str): arguments given to the script.

    Returns:
        parsed args
    """
    parser = argparse.ArgumentParser(description='Make the ridge plot')

    parser.add_argument('--mlruns_dir', type=str, required=True,
                        help='Top level path to the results')

    args = parser.parse_args(args)

    return args


def make_ridge_plot(mlruns_dir):
    all_exp_dirs = glob.glob(os.path.join(mlruns_dir, '*'))

    order = {'l': 0,
             'pa-121': 1,
             'pa-201': 2,
             'stacked': 3,
             'hemis': 4,
             'hemis-cl': 5,
             'dualnet': 6,
             'multitask': 7,
             'multitask-cl': 8}

    results = {
        'l': [],
        'pa-121': [],
        'pa-201': [],
        'stacked': [],
        'hemis': [],
        'hemis-cl': [],
        'dualnet': [],
        'multitask': [],
        'multitask-cl': []
    }
    exp_res = []
    value_res = []
    order_res = []

    for exp_dir in all_exp_dirs:
        with open(os.path.join(exp_dir, 'meta.yaml'), 'r') as f:
            meta = yaml.safe_load(f)

        if not meta['name'].startswith('lateral-view'):
            continue

        current_exps_dirs = glob.glob(os.path.join(exp_dir, '*'))
        for current_exp_dir in current_exps_dirs:
            if not os.path.isdir(current_exp_dir):
                continue

            # We don't want the best model
            runname_path = os.path.join(current_exp_dir, 'tags/mlflow.runName')
            with open(runname_path, 'r') as f:
                run_name = f.read()
            if 'best' in run_name:
                continue

            # auc must exist
            auc_path = os.path.join(current_exp_dir, 'metrics/auc')
            if not os.path.exists(auc_path):
                continue

            with open(auc_path, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                auc = np.array([[float(c) for c in row] for row in reader])

            max_auc = auc[:, 1].max()
            exp_name = meta['name'][13:]

            if exp_name == 'pa':
                # Sort by architecture
                if run_name.startswith('densenet'):
                    exp_name = f'{exp_name}-{run_name[8:11]}'
                else:
                    continue
            elif exp_name == 'hemis':
                # Sort by date
                with open(os.path.join(current_exp_dir, 'meta.yaml'), 'r') as f:
                    exp_meta = yaml.safe_load(f)
                start_time = datetime.datetime.utcfromtimestamp(int(exp_meta['start_time']) // 1000)
                threshold_time = datetime.datetime(2020, 1, 26)

                if start_time > threshold_time:
                    exp_name = f'{exp_name}-cl'
            elif exp_name == 'multitask':
                # Sort by mt-task-prob
                task_prob_path = os.path.join(current_exp_dir, 'params/mt-task-prob')
                if not os.path.exists(task_prob_path):
                    continue
                with open(task_prob_path, 'r') as f:
                    task_prob = f.read()

                if task_prob != '0.0':
                    exp_name = f'{exp_name}-cl'

            results[exp_name].append(max_auc)
            exp_res.append(exp_name)
            value_res.append(max_auc)
            order_res.append(order[exp_name])

    for k, v in results.items():
        print(k, len(v), np.mean(v), np.std(v))

    df = pd.DataFrame(dict(AUC=value_res, g=exp_res, o=order_res))
    df = df.sort_values(by=['o'])

    # Plotting
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

    g.map(sns.kdeplot, "AUC", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.002)
    g.map(sns.kdeplot, "AUC", clip_on=False, color="w", lw=2, bw=.002)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, "AUC")

    g.fig.subplots_adjust(hspace=-.25)

    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    make_ridge_plot(args.mlruns_dir)
