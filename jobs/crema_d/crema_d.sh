#!/bin/bash

#SBATCH --job-name=vjepa-crema-d
#SBATCH --output=/mnt/slurm/lennart/jepaslt/logs/%x-%j.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --nodes=1

# Activating mamba virtual env
echo "Activating mamba virtual environment."
source /mnt/data/miniconda3/bin/activate

# Creating environment if it does not already exist
env_yml_path=/mnt/slurm/lennart/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/mnt/data/miniconda/envs/$env_name

conda env create --debug --file=$env_yml_path
conda activate $env_name

export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt

# The splits used here must be created manually with the crema_d_utils.py utility.

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_0_16x8x3.yaml --devices cuda:0
python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_1_16x8x3.yaml --devices cuda:0
python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_2_16x8x3.yaml --devices cuda:0
python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_3_16x8x3.yaml --devices cuda:0
python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_crema_d_split_4_16x8x3.yaml --devices cuda:0

# I also need to "unexport" the path, because otherwise I will clutter this env variable.
export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/}
conda deactivate
conda env remove -n $env_name -y