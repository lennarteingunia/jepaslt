#!/bin/bash

#SBATCH --job-name=vjepa-ravdess-full-video-eval
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/logs/%j-%x.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a40:3
#SBATCH --mem=40G
#SBATCH --nodes=1

# Activating conda

echo "Activating mamba virtual environment."
source /mnt/data/miniconda3/bin/activate

# Creating environment if it does not already exist
env_yml_path=/mnt/slurm/lennart/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/mnt/data/miniconda/envs/$env_name

conda env update -n $env_name --file=$env_yml_path --prune
conda activate $env_name

python -m pip install pysha3

# Actually run the evaluation script

CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/src python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_ravdess_split_0_pretrained_full_video.yaml --devices cuda:0 cuda:1 cuda:2
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/src python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_ravdess_split_1_pretrained_full_video.yaml --devices cuda:0 cuda:1 cuda:2
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/src python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_ravdess_split_2_pretrained_full_video.yaml --devices cuda:0 cuda:1 cuda:2
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/src python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_ravdess_split_3_pretrained_full_video.yaml --devices cuda:0 cuda:1 cuda:2
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/src python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_ravdess_split_4_pretrained_full_video.yaml --devices cuda:0 cuda:1 cuda:2
