#!/bin/bash
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH -o /network/tmp1/bertranh/padchest/d666/slurm-%j.out

# 1. Load your environment
source activate py3

# 2. Copy your dataset on the compute node
# rsync -avz /network/data/<dataset> $SLURM_TMPDIR

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd ~/dev/pc-hemis/
python -u train.py '/network/data1/academictorrents-datastore/PADCHEST_SJ/processed' '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/joint_PA_L.csv' '/network/data1/academictorrents-datastore/PADCHEST_SJ/labels_csv/splits_PA_L_{}.pkl' '/network/tmp1/bertranh/padchest/d{}/' --target $1 --batch_size 8 --learning_rate 0.0001 --min_patients 100 --seed $2

# 4. Copy whatever you want to save on $SCRATCH
# rsync -avz $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
