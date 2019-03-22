from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image
import numpy as np
import pandas as pd
import os
from os.path import join


class PCXRay(Dataset):
    def __init__(self, datadir, csvpath):

        self.datadir = datadir
        self.df = pd.read_csv(csvpath)

        self._build_labels()
        self.mb = MultiLabelBinarizer(classes=self.labels)
        self.mb.fit(self.labels)
        
    def __len__(self):
        return len(self.df.PatientID.unique())

    def __getitem__(self, idx):
        subset = self.df[self.df.PatientID == self.df.PatientID[idx * 2]]
        labels = eval(subset.Clean_Labels.tolist()[0])
        encoded_labels = self.mb.transform([labels]).squeeze()

        pa_path = subset[subset.Projection == 'PA'][['ImageID', 'ImageDir']]
        pa_path = join(self.datadir, str(int(pa_path['ImageDir'].tolist()[0])), pa_path['ImageID'].tolist()[0])
        pa_path = './data/imgs/46523715740384360192496023767246369337_veyewt.png'  # TODO remove
        pa_img = np.array(Image.open(pa_path))

        l_path = subset[subset.Projection == 'L'][['ImageID', 'ImageDir']]
        l_path = join(self.datadir, str(int(l_path['ImageDir'].tolist()[0])), l_path['ImageID'].tolist()[0])
        l_path = './data/imgs/46523715740384360192496023767246369337_veyewt.png'  # TODO remove
        l_img = np.array(Image.open(l_path))

        return {'PA': pa_img, 'L': l_img, 'labels': labels, 'encoded_labels': encoded_labels}

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
            if v > 100:
                labels.append(k)
                labels_count.append(v)

        self.labels = labels
        self.labels_count = labels_count
        self.nb_labels = len(self.labels)


if __name__ == '__main__':
    cohort_file = './data/cxr8_joint_cohort_data.csv'
    img_dir = './data/imgs'

    dataset = PCXRay(img_dir, cohort_file)

    sample = dataset[0]
    print(sample['PA'].max())
