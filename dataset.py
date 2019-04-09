from torch.utils.data import Dataset
from sklearn.preprocessing import MultiLabelBinarizer
import torch

from PIL import Image
import numpy as np
import pandas as pd
from os.path import join
import pickle


def split_dataset(csvpath, output, train=0.6, val=0.2, seed=666):
    df = pd.read_csv(csvpath)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    patients_ids = df.PatientID.unique()

    train_val_split_idx = int(train * len(patients_ids))
    val_test_split_idx = int((train + val) * len(patients_ids))

    train_ids = patients_ids[:train_val_split_idx]
    val_ids = patients_ids[train_val_split_idx:val_test_split_idx]
    test_ids = patients_ids[val_test_split_idx:]

    with open(output, 'wb') as f:
        pickle.dump((train_ids, val_ids, test_ids), f)


class PCXRayDataset(Dataset):
    def __init__(self, datadir, csvpath, splitpath, transform=None, dataset='train', pretrained=False):
        super(PCXRayDataset, self).__init__()

        assert dataset in ['train', 'val', 'test']

        self.datadir = datadir
        self.transform = transform
        self.pretrained = pretrained

        self.df = pd.read_csv(csvpath)

        self._build_labels()
        self.mb = MultiLabelBinarizer(classes=self.labels)
        self.mb.fit(self.labels)

        # Split into train or validation
        with open(splitpath, 'rb') as f:
            train_ids, val_ids, test_ids = pickle.load(f)
        if dataset == 'train':
            self.df = self.df[self.df.PatientID.isin(train_ids)]
        elif dataset == 'val':
            self.df = self.df[self.df.PatientID.isin(val_ids)]
        else:
            self.df = self.df[self.df.PatientID.isin(test_ids)]

        self.df = self.df.reset_index()
        
    def __len__(self):
        return len(self.df.PatientID.unique())

    def __getitem__(self, idx):
        subset = self.df[self.df.PatientID == self.df.PatientID[idx * 2]]
        labels = eval(subset.Clean_Labels.tolist()[0])
        labels = [l for l in labels if l in self.labels]
        encoded_labels = self.mb.transform([labels]).squeeze()

        pa_path = subset[subset.Projection == 'PA'][['ImageID', 'ImageDir']]
        pa_path = join(self.datadir, str(int(pa_path['ImageDir'].tolist()[0])), pa_path['ImageID'].tolist()[0])
        # pa_path = './data/processed/0/46523715740384360192496023767246369337_veyewt.png'
        pa_img = np.array(Image.open(pa_path))[np.newaxis]
        if self.pretrained:
            pa_img = np.repeat(pa_img, 3, axis=0)

        l_path = subset[subset.Projection == 'L'][['ImageID', 'ImageDir']]
        l_path = join(self.datadir, str(int(l_path['ImageDir'].tolist()[0])), l_path['ImageID'].tolist()[0])
        # l_path = './data/processed/0/46523715740384360192496023767246369337_veyewt.png'
        l_img = np.array(Image.open(l_path))[np.newaxis]
        if self.pretrained:
            l_img = np.repeat(l_img, 3, axis=0)

        sample = {'PA': pa_img, 'L': l_img}

        if self.transform is not None:
            sample = self.transform(sample)

        sample['labels'] = labels
        sample['encoded_labels'] = torch.from_numpy(encoded_labels.astype(np.float32))
        sample['sample_weight'] = torch.max(sample['encoded_labels'] * self.labels_weights)

        return sample

    def _build_labels(self):
        labels_dict = {}
        for labels in self.df.Clean_Labels:
            for label in eval(labels):
                label = label.strip()
                if label not in labels_dict:
                    labels_dict[label] = 0
                labels_dict[label] += 1

        labels = []
        labels_count = []
        for k, v in labels_dict.items():
            if v > 1000:
                labels.append(k)
                labels_count.append(v)

        self.labels = labels
        self.labels_count = labels_count
        self.labels_weights = torch.from_numpy(np.array([(len(self) / label)
                                                         for label in labels_count], dtype=np.float32))
        self.labels_weights = torch.clamp(self.labels_weights * 0.1, 1., 5.)
        self.nb_labels = len(self.labels)


class Normalize(object):
    """
    Changes images values to be between -1 and 1.
    """

    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        pa_img = 2 * (pa_img / 65536) - 1.
        pa_img = pa_img.astype(np.float32)
        l_img = 2 * (l_img / 65536) - 1.
        l_img = l_img.astype(np.float32)

        return {'PA': pa_img, 'L': l_img}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        return {'PA': torch.from_numpy(pa_img),
                'L': torch.from_numpy(l_img)}


if __name__ == '__main__':
    cohort_file = './data/cxr8_joint_cohort_data.csv'
    img_dir = './data/processed'
    split_file = './models/data_split.pkl'

    split_dataset(cohort_file, split_file)
    dataset = PCXRayDataset(img_dir, cohort_file, split_file)
    print(dataset.labels_weights)
    print(dataset.labels_count)
    for i in range(100):
        print(dataset[i]['sample_weight'])
