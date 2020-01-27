#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=4G

DATADIR=/lustre04/scratch/cohenjos/PC/images-224/
LABELSDIR=$HOME/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/

source $HOME/homer/bin/activate
cd ~/works/lateral-view-analysis/

python extract_cohort.py --input_csv $LABELSDIR/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv \
--output_csv $LABELSDIR/PA_only.csv --datadir $DATADIR --broken-file $LABELSDIR/broken_images \
--mode 'pa' --joint-csv $LABELSDIR/joint_PA_L.csv 



# python -u preprocess_images.py '/network/data1/academictorrents-datastore/PADCHEST_SJ/image_zips/' '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/'
# python -u extract_cohort.py '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/PADCHEST_chest_x_ray_images_labels_160K.csv' '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/joint_PA_L.csv' '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/' -b '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed/broken_images.txt'
