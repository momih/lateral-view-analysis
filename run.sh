#!/bin/bash
hostname
export LANG=C.UTF-8
source $HOME/.bashrc

DATADIR=~/scratch/PC/images-224
#DATADIR=$SLURM_TMPDIR/images-224
#rsync -a --info=progress2 images-224.tar $SLURM_TMPDIR/
#time tar xf $SLURM_TMPDIR/images-224.tar -C $SLURM_TMPDIR/ --strip=4

python3 -u train.py $DATADIR ~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/joint_PA_L.csv ~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_{}.pkl /lustre04/scratch/cohenjos/PC-output/joe-pa-densenet-s{} --batch_size=64 $@



