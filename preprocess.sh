#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G

source activate py3

cd ~/dev/pc-hemis/
# python -u preprocess_images.py '/network/data1/academictorrents-datastore/PADCHEST_SJ/image_zips/' '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/'
python -u extract_cohort.py '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/PADCHEST_chest_x_ray_images_labels_160K.csv' '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/joint_PA_L.csv' '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/' -b '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/broken_images.txt'
