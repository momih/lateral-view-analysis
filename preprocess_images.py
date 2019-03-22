"""
Extract the images of the StudyIDs in the cohort, then center crop and resize
to (224,224) and save.
"""

from PIL import Image, ImageOps
from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
import os


def preprocess_images(img_dir, target_dir, csvpath):
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)

    data = pd.read_csv(csvpath)
    data.ImageDir = data.ImageDir.apply(lambda x: "{}.zip".format(int(x)))
    grouped = data.groupby('ImageDir').ImageID.apply(list).to_dict()

    for zipped_file, image_list in grouped.items():
        with ZipFile(img_dir + zipped_file) as archive:
            for img in image_list:
                with archive.open(img) as file:
                    im = Image.open(file)
                    im = ImageOps.fit(im, size=(224, 224))
                    dest = os.path.join(img_dir + img)
                    im.save(dest)


if __name__ == '__main__':
    cohort_file = './data/PADCHEST_chest_x_ray_images_labels_160K.csv'
    img_dir = './data/imgs'
    target_dir = './data/preprocessed'

    preprocess_images(img_dir, target_dir, cohort_file)
