from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

from PIL import Image
import numpy as np
import pandas as pd
import os

class PCXRay(Dataset):
    def __init__(self, datadir, csvpath):

        self.datadir = datadir
        self.full_df = pd.read_csv(csvpath)
        df = self.full_df.drop_duplicates(subset=['StudyID', 'Projection'])
        df = df.pivot(index='StudyID', columns='Projection', values=['ImageID', 'CXR_Label'])
        df.columns = ["_".join(x) for x in df.columns.to_flat_index()]
        self.data = df.drop(columns='CXR_Label_L').to_dict('index')
        self.idx_to_study = {k:v for k,v in enumerate(self.data.keys())}
        
        labels = ['pneumonia', 'effusion', 'consolidation', 'no finding', 'cardiomegaly', 
                  'infiltration', 'emphysema', 'mass', 'hernia', 'atelectasis', 
                  'pneumothorax', 'edema', 'pleural thickening', 'nodule', 'fibrosis']
        self.mb = MultiLabelBinarizer(classes=labels)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        study_data = self.data[self.idx_to_study[idx]]
        xrays = self._read_imgs(study_data)
        labels = study_data['CXR_Label_PA'].split("|")
        labels =  self.mb.fit_transform([labels])
        return xrays, labels
    
    def _read_imgs(self, study):
        img_list = []
        for view in ['PA', 'L']:
            img_path = os.path.join(self.datadir, study['ImageID_' + view])
            img = np.array(Image.open(img_path))
            img_list.append(img)
        return np.stack(img_list)
        