from tqdm import tqdm
import pandas as pd
import random
import re


def get_cohort(output):
    tqdm.pandas()
    random.seed(9999)

    usecols = ['ImageID', 'ImageDir', 'StudyDate_DICOM', 'StudyID', 'PatientID',
               'Projection', 'Pediatric', 'Rows_DICOM', 'Columns_DICOM', 'Labels']
    df = pd.read_csv('./data/PADCHEST_chest_x_ray_images_labels_160K.csv', usecols=usecols,
                     low_memory=False)

    print("{0} images in dataset.".format(len(df)))

    # Only keeping those images that are L or PA and removing pediatric patients
    df = df.loc[(df.Projection.isin(['L', 'PA'])) & (df.Pediatric == 'No')]

    # Removing duplicates
    df = df.drop_duplicates(subset=df.columns[1:]).drop(columns=['Pediatric'])

    # Removing images that don't have labels
    df = df.dropna(subset=['Labels'])

    # Keeping only those IDS that have both PA and L
    projs = df.groupby('StudyID').Projection.progress_apply(lambda x: ",".join(x))
    ids_to_keep = projs[projs.isin(['PA,L', 'L,PA'])].index

    df = df.loc[df.StudyID.isin(ids_to_keep)]

    # Keeping only the first study for a patient
    projs = df.groupby('PatientID').StudyDate_DICOM.progress_apply(lambda x: x.min())
    ids_to_keep = df.apply(lambda x: x.StudyDate_DICOM == projs[x.PatientID], axis=1)
    df = df.loc[ids_to_keep]

    # Some patients had multiple studies done the same day, so we pick randomly among those
    projs = df.groupby('PatientID').StudyID.progress_apply(lambda x: random.choice(x.tolist()))
    ids_to_keep = df.apply(lambda x: x.StudyID == projs[x.PatientID], axis=1)
    df = df.loc[ids_to_keep]

    # Removing images whose label is 'exclude' or 'suboptimal study'
    labels_to_remove = re.compile('exclude|suboptimal study|unchanged')

    def remove_bad_labels(x):
        match = set(labels_to_remove.findall(x))
        return False if match else True

    good_labels = df.Labels.apply(remove_bad_labels)
    df = df.loc[good_labels]

    df.to_csv(output, index=False)

    print("{0} images in cohort from {1} patients.".format(len(df), len(df.PatientID.unique())))
    check_study_patient = (len(df.StudyID.unique()) == len(df.PatientID.unique()))
    check_study_image = (len(df.StudyID.unique()) * 2 == len(df))
    print("Check: {} {}".format(check_study_patient, check_study_image))


def labels_distribution(cohort):
    tqdm.pandas()

    df = pd.read_csv(cohort, low_memory=False)

    labels_dict = {}
    for labels in df.Labels:
        for label in eval(labels):
            label = label.strip()
            if label in ['', 'chronic changes']:
                continue
            if label not in labels_dict:
                labels_dict[label] = 0
            labels_dict[label] += 1

    for k, v in sorted(labels_dict.items(), key=lambda x: x[1], reverse=True):
        if v > 500:
            print(k, v)


if __name__ == '__main__':
    cohort_file = './data/cxr8_joint_cohort_data.csv'

    get_cohort(cohort_file)
    labels_distribution(cohort_file)
