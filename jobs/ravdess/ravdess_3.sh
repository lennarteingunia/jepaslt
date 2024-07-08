#!/bin/bash

#SBATCH --job-name=vjepa-ravdess
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/jepaslt/.logs/%x-%j/slurm.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --nodes=1

# Activating conda

source /mnt/data/miniconda3/bin/activate

# Creating conda environment if it does not already exists

env_yml_path=/mnt/slurm/lennart/jepaslt/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/mnt/data/miniconda3/envs/$env_name

if [ ! -d $env_path ]
then
	echo "Specified environment does not exist. CREATING..."
	conda env create --debug --file=$env_yml_path
else
	echo "Specified environment already exists. UPDATING..."
	conda env update -n $env_name --debug --file=$env_yml_path --prune
fi

# Activating the environment

conda activate $env_name

# Actually run the evaluation script

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_3.yaml --devices cuda:0

# I also need to "unexport" the path, because otherwise I will clutter this env variable.
export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/jepaslt/}