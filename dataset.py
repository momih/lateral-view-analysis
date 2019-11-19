import logging
import pickle
from os.path import join

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_dataset(csvpath: str, output: str, train=0.6, val=0.2, seed=666) -> None:
    """
    Split the data contained in csvpath in train/val/test, and write the results in output.
    """
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
    def __init__(self, datadir, csvpath, splitpath, transform=None,
                 dataset='train', pretrained=False, min_patients_per_label=50,
                 exclude_labels=["other", "normal", "no finding"], flat_dir=True):
        """
        Data reader. Only selects labels that at least min_patients_per_label patients have.
        """
        super(PCXRayDataset, self).__init__()

        assert dataset in ['train', 'val', 'test']

        self.datadir = datadir
        self.transform = transform
        self.pretrained = pretrained
        self.threshold = min_patients_per_label
        self.exclude_labels = exclude_labels
        self.flat_dir = flat_dir
        self.df = pd.read_csv(csvpath)

        self._build_labels()
        self.mb = MultiLabelBinarizer(classes=self.labels)
        self.mb.fit(self.labels)

        # Split into train or validation
        if splitpath is not None:
            with open(splitpath, 'rb') as f:
                train_ids, val_ids, test_ids = pickle.load(f)
            if dataset == 'train':
                self.df = self.df[self.df.PatientID.isin(train_ids)]
            elif dataset == 'val':
                self.df = self.df[self.df.PatientID.isin(val_ids)]
            else:
                self.df = self.df[self.df.PatientID.isin(test_ids)]

            self.df = self.df.reset_index()

        self.df = self.df.sort_values('PatientID').reset_index(drop=True)

        def processdf(subset, to_keep):
            imageid = dict(zip(subset.Projection, subset.ImageID))
            imagedir = dict(zip(subset.Projection, subset.ImageDir))

            labels = eval(subset.Clean_Labels.tolist()[0])
            labels = list(set(labels).intersection(to_keep))

            return {'ImageDir': imagedir, 'ImageID': imageid, 'Labels': labels}

        self.metadata = self.df.groupby('PatientID').apply(lambda x: processdf(x, self.labels)).to_dict()
        self.idx2pt = {idx: x for idx, x in enumerate(self.df.PatientID.unique())}

    @property
    def targets(self):
        targets = [self.metadata[pt]['Labels'] for pt in self.idx2pt.values()]
        return self.mb.transform(targets)

    @property
    def data(self):
        files = []
        for pt in self.idx2pt.values():
            data = self.metadata[pt]
            pa_dir = str(int(data['ImageDir']['PA'])) if not self.flat_dir else ''
            pa_path = join(self.datadir, pa_dir, data['ImageID']['PA'])
            files.append(pa_path)

        logger.info("Reading files")
        imgs = np.stack([np.array(Image.open(path)) for path in tqdm(files)])
        imgs = np.expand_dims(imgs, -1)
        return imgs

    def __len__(self):
        return len(self.df.PatientID.unique())

    def __getitem__(self, idx):

        pt_id = self.idx2pt[idx]
        data = self.metadata[pt_id]

        labels = data['Labels']
        encoded_labels = self.mb.transform([labels]).squeeze()

        pa_dir = str(int(data['ImageDir']['PA'])) if not self.flat_dir else ''
        pa_path = join(self.datadir, pa_dir, data['ImageID']['PA'])
        pa_img = np.array(Image.open(pa_path))[..., np.newaxis]

        l_dir = str(int(data['ImageDir']['L'])) if not self.flat_dir else ''
        l_path = join(self.datadir, l_dir, data['ImageID']['L'])
        l_img = np.array(Image.open(l_path))[..., np.newaxis]

        if self.pretrained:
            pa_img = np.repeat(pa_img, 3, axis=-1)
            l_img = np.repeat(l_img, 3, axis=-1)

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
            if k in self.exclude_labels:
                logger.info("excluding label {} which occured {} times".format(k, v))
                continue
            if v > self.threshold * 2:
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

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        to_tensor = transforms.ToTensor()
        sample['PA'] = to_tensor(sample['PA'])
        sample['L'] = to_tensor(sample['L'])

        return sample


class ToPILImage(object):
    """
    Convert ndarrays in sample to PIL images.
    """

    def __call__(self, sample):
        to_pil = transforms.ToPILImage()
        sample['PA'] = to_pil(sample['PA'])
        sample['L'] = to_pil(sample['L'])

        return sample


class GaussianNoise(object):
    """
    Adds Gaussian noise to the PA and L (mean 0, std 0.05)
    """

    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']

        pa_img += torch.randn_like(pa_img) * 0.05
        l_img += torch.randn_like(l_img) * 0.05

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


class RandomRotation(object):
    """
    Adds a random rotation to the PA and L.
    """

    def __init__(self, degrees=5):
        self.rot = transforms.RandomRotation(degrees=degrees)

    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']
        pa_img = self.rot(pa_img)
        l_img = self.rot(l_img)

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


class RandomTranslate(object):
    """
    Adds a random translation to the PA and L.
    """

    def __init__(self, translate=None):
        self.aff = transforms.RandomAffine(degrees=0, translate=translate)

    def __call__(self, sample):
        pa_img, l_img = sample['PA'], sample['L']
        pa_img = self.aff(pa_img)
        l_img = self.aff(l_img)

        sample['PA'] = pa_img
        sample['L'] = l_img
        return sample


if __name__ == '__main__':
    cohort_file = './data/cxr8_joint_cohort_data.csv'
    img_dir = './data/processed'
    split_file = './models/data_split.pkl'

    split_dataset(cohort_file, split_file)
    dataset = PCXRayDataset(img_dir, cohort_file, split_file)
    logger.info(dataset.labels_weights)
    logger.info(dataset.labels_count)
    for i in range(100):
        logger.info(dataset[i]['sample_weight'])
