#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=2
#SBATCH --array=1-20%5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=5:00:00
#SBATCH --job-name=orion_lateral
#SBATCH --output=logs/out_%a.log
#SBATCH --error=logs/err_%a.log

# 1. Create your environment
module load python/3.7.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cp /home/hrb/dev/lateral-view-analysis/requirements.txt $SLURM_TMPDIR/requirements.txt
sed -i '1 i\-f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic' $SLURM_TMPDIR/requirements.txt
sed -i '1 i\-f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/avx2' $SLURM_TMPDIR/requirements.txt
sed -i '1 i\-f /cvmfs/soft.computecanada.ca/custom/python/wheelhouse/avx512' $SLURM_TMPDIR/requirements.txt
sed -i '1 i\-f /home/hrb/.wheels' $SLURM_TMPDIR/requirements.txt

pip install --no-index -r $SLURM_TMPDIR/requirements.txt

export ORION_DB_ADDRESS='/home/hrb/dev/lateral-view-analysis/orion.pkl'
export ORION_DB_TYPE='pickleddb'
export ORION_DB_NAME='lateral_view_analysis'

# 2. Copy your dataset on the compute node
export DATADIR=$SLURM_TMPDIR/images-224
time rsync -a --info=progress2 /lustre04/scratch/cohenjos/PC/images-224.tar $SLURM_TMPDIR/
time tar xf $SLURM_TMPDIR/images-224.tar -C $SLURM_TMPDIR/ --strip=4

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd ~/dev/lateral-view-analysis/
orion -v hunt --config orion_config.yaml ./train.py 'CLUSTER' ~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/joint_PA_L.csv ~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_666.pkl /lustre04/scratch/cohenjos/PC-output/hadrien-test-will-be-removed --exp_name {trial.id} --seed 666 --epochs 20 --model-type 'dualnet' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1.0)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim "orion~choices(['adam', 'sgd'])" --reduce_period 'orion~uniform(5, 30, discrete=True)' --gamma 'orion~uniform(0.1, 0.5)' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# 4. Copy whatever you want to save on $SCRATCH
# rsync -avz $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
