#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=2
#SBATCH --array=1-35%5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=orion_lateral
#SBATCH --output=logs/out_%A_%a.log

# 1. Create your environment
module load python/3.7.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip

cp /home/hrb/dev/lateral-view-analysis/requirements.txt $SLURM_TMPDIR/requirements.txt
sed -i '1 i\-f /home/hrb/.wheels' $SLURM_TMPDIR/requirements.txt

pip install --no-index -r $SLURM_TMPDIR/requirements.txt

export ORION_DB_ADDRESS='/home/hrb/dev/lateral-view-analysis/orion.pkl'
export ORION_DB_TYPE='pickleddb'
export ORION_DB_NAME='lateral_view_analysis'

# 2. Copy your dataset on the compute node
export DATADIR=$SLURM_TMPDIR/images-224
time rsync -a --info=progress2 /lustre04/scratch/cohenjos/PC/images-224.tar $SLURM_TMPDIR/
time tar xf $SLURM_TMPDIR/images-224.tar -C $SLURM_TMPDIR/

# 3. Launch your job, tell it to save the model in $SLURM_TMPDIR
#    and look for the dataset into $SLURM_TMPDIR
cd ~/dev/lateral-view-analysis/
DATADIRVAR='CLUSTER'
CSV=~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/joint_PA_L.csv
SPLIT=~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_666.pkl
OUTPUT=/lustre04/scratch/cohenjos/PC-output/hadrien
EPOCHS=40
SEED=666

# PA DenseNet121
# orion -v hunt -n lateral-view-pa8 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --arch 'densenet121' --target 'pa' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=(1,))' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# PA DenseNet201
# orion -v hunt -n lateral-view-pa --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --arch 'densenet201' --target 'pa' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=(1,))' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# PA ResNet152
# orion -v hunt -n lateral-view-pa --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --arch 'resnet152' --target 'pa' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=(1,))' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# L DenseNet121
# orion -v hunt -n lateral-view-l4 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --target 'l' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=(1,))' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Stacked
# orion -v hunt -n lateral-view-stacked --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'stacked' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Hemis
# orion -v hunt -n lateral-view-hemis --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'hemis' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Hemis CL
# orion -v hunt -n lateral-view-hemis3 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'hemis' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --drop-view-prob 'orion~choices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# DualNet
# orion -v hunt -n lateral-view-dualnet2 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Multitask
orion -v hunt -n lateral-view-multitask3 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'multitask' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --mt-task-prob 0.0 --mt-join "orion~choices(['concat', 'max', 'mean'])" --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Multitask CL
# orion -v hunt -n lateral-view-multitask --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'multitask' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --mt-task-prob 'orion~choices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])' --mt-join "orion~choices(['concat', 'max', 'mean'])" --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# 4. Copy whatever you want to save on $SCRATCH
# rsync -avz $SLURM_TMPDIR/<to_save> /network/tmp1/<user>/
