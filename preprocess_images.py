"""
Extract the images of the StudyIDs in the cohort, then center crop and resize
to (224,224) and save.
"""

from PIL import Image, ImageOps
from zipfile import ZipFile
import pandas as pd
from tqdm import tqdm
import os

SOURCEDIR = './data/imgs/'
TARGETDIR = './data/subset/'

if not os.path.exists(TARGETDIR):
    os.mkdir(TARGETDIR)
    
data = pd.read_csv('./data/cohort_data.csv')
data.ImageDir = data.ImageDir.apply(lambda x: "{}.zip".format(int(x)))
grouped = data.groupby('ImageDir').ImageID.apply(list).to_dict()

for zipped_file, image_list in grouped.items():
    with ZipFile(SOURCEDIR + zipped_file) as archive:
        for img in image_list:
            with archive.open(img) as file:
                im = Image.open(file)
                im = ImageOps.fit(im, size=(224,224))
                dest = os.path.join(TARGETDIR + img)
                im.save(dest)
