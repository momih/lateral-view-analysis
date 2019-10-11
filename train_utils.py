import numpy as np
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate_model(model, dataloader, loss_fn=None, target='joint', model_type=None, vote_at_test=False):
    with torch.no_grad():
        runningloss = torch.zeros(1, requires_grad=False, dtype=torch.float).to(DEVICE)
        y_preds, y_true = [], []
        for data in dataloader:
            if target == 'joint':
                *images, label = data['PA'].to(DEVICE), data['L'].to(DEVICE), data['encoded_labels'].to(DEVICE)
                if model_type == 'stacked':
                    images = torch.cat(images, dim=1)
            else:
                images, label = data[target.upper()].to(DEVICE), data['encoded_labels'].to(DEVICE)

            # Forward
            output = model(images)
            if model_type == 'multitask':
                if vote_at_test:
                    output = torch.stack(output, dim=1).mean(dim=1)
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
    return y_true, y_preds, runningloss