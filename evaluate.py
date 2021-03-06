import pickle
from os.path import join

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModelEvaluator:
    def __init__(self, output_dir, target, logger, averaging='macro'):
        self.output_dir = output_dir
        self.target = target
        self.logger = logger
        self.averaging = averaging

        self.store_dict = {'train_loss': [], 'val_loss': [], 'val_preds_all': [], 'val_auc': [], 'val_prc': []}
        self.eval_df = pd.DataFrame(columns=['epoch', 'avg_acc', 'pure_acc', 'auc', 'prc', 'loss'])

    def load_saved(self):
        for metric in self.store_dict.keys():
            with open(join(self.output_dir, f'{self.target}-{metric}.pkl'), 'rb') as f:
                self.store_dict[metric] = pickle.load(f)
        self.eval_df = pd.read_csv(join(self.output_dir, f'{self.target}-metrics.csv'))

    def evaluate_and_save(self, y_true, y_pred, epoch, train_true=None, train_preds=None, runloss=None):
        val_auc = roc_auc_score(y_true, y_pred, average=None)
        val_prc = average_precision_score(y_true, y_pred, average=None)

        self.logger.info("Validation AUC, Train AUC and difference")
        try:
            train_auc = roc_auc_score(train_true, train_preds, average=None)
        except:
            self.logger.info('Error in calculating train AUC')
            train_auc = np.zeros_like(val_auc)

        # diff_train_val = val_auc - train_auc
        # diff_train_val = np.stack([val_auc, train_auc, diff_train_val], axis=-1)
        # self.logger.info(diff_train_val.round(4))

        self.store_dict['val_prc'].append(val_prc)
        self.store_dict['val_preds_all'].append(y_pred)
        self.store_dict['val_auc'].append(val_auc)
        self.store_dict['val_loss'].append(runloss)

        for metric in self.store_dict.keys():
            with open(join(self.output_dir, '{}-{}.pkl'.format(self.target, metric)), 'wb') as f:
                pickle.dump(self.store_dict[metric], f)

        y_pred_binary = np.where(y_pred > 0.5, 1, 0)
        avg_acc_per_label = []
        for i in range(y_true.shape[1]):
            acc_of_label = accuracy_score(y_true[:, i], y_pred_binary[:, i])
            avg_acc_per_label.append(acc_of_label)
        avg_acc_per_label = np.mean(avg_acc_per_label)

        metrics = {'epoch': epoch + 1,
                   'pure_acc': accuracy_score(y_true, y_pred_binary),
                   'avg_acc': avg_acc_per_label,
                   'auc': roc_auc_score(y_true, y_pred, average=self.averaging),
                   'prc': average_precision_score(y_true, y_pred, average=self.averaging),
                   'loss': runloss}
        self.logger.info(metrics)

        for k, v in metrics.items():
            mlflow.log_metric(k, v, step=epoch)

        self.eval_df = self.eval_df.append(metrics, ignore_index=True)
        self.eval_df.to_csv(join(self.output_dir, '{}-metrics.csv'.format(self.target)), index=False)
        return val_auc, val_prc


def get_model_preds(model, dataloader, loss_fn=None, target='joint', test_on='both',
                    model_type=None, vote_at_test=False, progress_bar=False):
    with torch.no_grad():
        runningloss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(DEVICE)
        y_preds, y_true = [], []

        if progress_bar:
            dataloader = tqdm(dataloader)

        for data in dataloader:
            if target == 'joint':
                *images, label = data['PA'].to(DEVICE), data['L'].to(DEVICE), data['encoded_labels'].to(DEVICE)

                if test_on in ['pa', 'l']:
                    pa, l = images
                    images = [torch.zeros_like(l), l] if test_on == 'l' else [pa, torch.zeros_like(pa)]

                if model_type == 'stacked':
                    images = torch.cat(images, dim=1)
            else:
                images, label = data[target.upper()].to(DEVICE), data['encoded_labels'].to(DEVICE)

            # Forward
            output = model(images)
            if model_type == 'multitask':
                if vote_at_test:
                    output = torch.stack(output, dim=1).mean(dim=1)
                elif test_on in ['pa', 'l']:
                    output = output[2] if test_on == 'l' else output[1]
                else:
                    output = output[0]

            if loss_fn is not None:
                runningloss += loss_fn(output, label).mean().detach().data

            # Save predictions
            y_preds.append(torch.sigmoid(output).detach().cpu().numpy())
            y_true.append(label.detach().cpu().numpy())
            del output, images, data

    y_true = np.vstack(y_true)
    y_preds = np.vstack(y_preds)
    runningloss = runningloss.item()

    return y_true, y_preds, runningloss
