#!/bin/bash
#SBATCH --account=rpp-bengioy
#SBATCH --cpus-per-task=2
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --job-name=orion_lateral
#SBATCH --output=logs/multiseed_%A_%a.log

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
SPLIT=~/projects/rpp-bengioy/jpcohen/PADCHEST_SJ/labels_csv/splits_PA_L_$SLURM_ARRAY_TASK_ID.pkl
OUTPUT=/lustre04/scratch/cohenjos/PC-output/hadrien
EPOCHS=40
SEED=$SLURM_ARRAY_TASK_ID

# PA DenseNet121
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_pa_121_s_{} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --arch 'densenet121' --target 'pa' --batch_size 8 --learning_rate ['0.000586158'] --dropout 0 --optim 'adam'

# PA DenseNet201
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_pa_201_s_{} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --arch 'densenet201' --target 'pa' --batch_size 8 --learning_rate ['0.000125313'] --dropout 1 --optim 'adam'

# L DenseNet121
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_l_121_s_{} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --target 'l' --batch_size 8 --learning_rate ['0.000268387'] --dropout 2 --optim 'adam'

# Stacked
./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_stacked_s_{} --seed $SEED --epochs $EPOCHS --model-type 'stacked' --target 'joint' --batch_size 8 --learning_rate "['0.000191423', '0.00010794319', '9.892179e-05']" --dropout 1 --optim 'adam'

# Hemis
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_hemis_s_{} --seed $SEED --epochs $EPOCHS --model-type 'hemis' --target 'joint' --batch_size 8 --learning_rate ['0.00037861971213', '1.965659e-05', '2.800382e-05'] --dropout 1 --optim 'adam'

# Hemis CL
# orion -v hunt -n lateral-view-hemis2 --config orion_config.yaml ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name {trial.id} --seed $SEED --epochs $EPOCHS --model-type 'hemis' --target 'joint' --batch_size 8 --learning_rate 'orion~loguniform(1e-5, 1e-3, shape=3)' --dropout 'orion~uniform(0, 5, discrete=True)' --optim 'adam' --mt-task-prob 'orion~choices([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])' --log '{exp.working_dir}/{exp.name}_{trial.id}/exp.log'

# Joint DualNet
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_dualnet_s_{} --seed $SEED --epochs $EPOCHS --model-type 'dualnet' --target 'joint' --batch_size 8 --learning_rate ['0.000298055004293', '0.00076431225830', '0.0002667413627'] --dropout 2 --optim 'adam'

# Joint Multitask
# ./hyperparam_search.py --data_dir $DATADIRVAR --csv_path $CSV --splits_path $SPLIT --output_dir $OUTPUT --exp_name best_multitask_s_{} --seed $SEED --epochs $EPOCHS --model-type 'multitask' --target 'joint' --batch_size 8 --learning_rate ['9.364812e-05', '3.8587835e-05', '1.29720523e-05'] --dropout 1 --optim 'adam' --mt-task-prob 0.1 --mt-join 'mean'
