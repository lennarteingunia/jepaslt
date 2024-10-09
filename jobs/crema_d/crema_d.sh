#!/bin/bash
#SBATCH --job-name=vjepa-crema-d
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/logs/%x-%j.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:0
#SBATCH --mem=40G
#SBATCH --nodes=1

# Activating mamba virtual env
echo "Activating mamba virtual environment."
source /mnt/data/miniconda3/bin/activate

# Creating environment if it does not already exist
env_yml_path=/mnt/slurm/lennart/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/mnt/data/miniconda/envs/$env_name

if [ ! -d $env_path ]
then
	echo "Specified environment does not exist. CREATING..."
	conda env create --debug --file=$env_yml_path
else
	echo "Specified environment already exists. UPDATING..."
	conda env update -n $env_name --debug --file=$env_yml_path --prune
fi

conda activate $env_name

# Syncing dataset
echo "RSyncing datasets..."
rsync -havzP --stats --delete /mnt/datasets/CREMA-D /mnt/data/ --exclude .git/

export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt
mkdir /mnt/data/CREMA-D/additional/splits/$SLURM_JOB_ID/
python crema_d_utils.py run /mnt/data/CREMA-D/ /mnt/data/CREMA-D/additional/splits/$SLURM_JOB_ID/

# The splits used here are created with the crema_d_utils.py utility.
CUDA_VISIBLE_DEVICES=0 python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_0_16x8x3.yaml --devices cuda:0
CUDA_VISIBLE_DEVICES=0 python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_1_16x8x3.yaml --devices cuda:0
CUDA_VISIBLE_DEVICES=0 python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_2_16x8x3.yaml --devices cuda:0
CUDA_VISIBLE_DEVICES=0 python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_3_16x8x3.yaml --devices cuda:0
CUDA_VISIBLE_DEVICES=0 python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_4_16x8x3.yaml --devices cuda:0