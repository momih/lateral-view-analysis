"""
For every image, center crop and resize to (224,224) and save.
"""
import argparse
from glob import glob
import os
from os.path import exists, join
from PIL import Image, ImageOps
from zipfile import ZipFile

from tqdm import tqdm


def preprocess_images(img_dir, target_dir):
    if not exists(target_dir):
        os.makedirs(target_dir)

    zip_list = glob(join(img_dir, '*.zip'))

    broken_file_list = []
    for zip_file in sorted(zip_list):
        target_zip_dir = zip_file.replace(img_dir, target_dir)[:-len('.zip')]
        if not exists(target_zip_dir):
            os.makedirs(target_zip_dir)

        with ZipFile(zip_file, 'r') as archive:
            for img_name in tqdm(archive.namelist(), desc=zip_file.replace(img_dir, '')):
                if not img_name.endswith('.png'):
                    continue

                with archive.open(img_name) as file:
                    try:
                        im = Image.open(file)
                        im = ImageOps.fit(im, size=(224, 224))
                    except Exception as e:
                        broken_file_list.append(img_name)
                        continue

                    im.save(join(target_zip_dir, img_name))

        with open(join(target_dir, 'broken_images.txt'), 'w') as f:
            for item in broken_file_list:
                f.write("{}\n".format(item))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Usage')
    parser.add_argument('img_dir', type=str)
    parser.add_argument('target_dir', type=str)
    args = parser.parse_args()

    preprocess_images(args.img_dir, args.target_dir)
