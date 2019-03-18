from tqdm import tqdm
import pandas as pd
import re

tqdm.pandas()

usecols = ['ImageID', 'ImageDir', 'StudyDate_DICOM', 'StudyID', 'PatientID',
           'Projection', 'Pediatric', 'Rows_DICOM', 'Columns_DICOM', 'Labels']
df = pd.read_csv('./data/PADCHEST_chest_x_ray_images_labels_160K.csv', usecols=usecols,
                 low_memory=False)

# Only keeping those images that are L or PA and removing pediatric patients
df = df.loc[(df.Projection.isin(['L', 'PA'])) & (df.Pediatric == 'No')]

# Removing duplicates
df = df.drop_duplicates(subset=df.columns[1:]).drop(columns=['Pediatric'])

# Removing images that don't have labels
df = df.dropna(subset=['Labels'])

projs = df.groupby('StudyID').Projection.progress_apply(lambda x: ",".join(set(x)))
ids_to_keep = projs[projs == 'PA,L'].index

# Keeping only those IDS that have both PA and L
df = df.loc[df.StudyID.isin(ids_to_keep)]

# converting the labels into chestxray format
cxr_labels = ['pneumonia', 'effusion', 'consolidation', 'no finding', 'cardiomegaly', 
              'infiltration', 'emphysema', 'mass', 'hernia', 'atelectasis', 
              'pneumothorax', 'edema', 'pleural thickening', 'nodule', 'fibrosis']
cxr8 = re.compile("|".join(cxr_labels))

def convert_to_cxr8_labels(x):
    match = set(cxr8.findall(x))
    new_label = "|".join(match) if match else 'no finding'
    return new_label

df['CXR_Label'] = df.Labels.apply(convert_to_cxr8_labels)
df.to_csv('./data/cohort_data.csv', index=False)
