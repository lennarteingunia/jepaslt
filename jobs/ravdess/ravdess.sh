#!/bin/bash

#SBATCH --job-name=vjepa-ravdess
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/jepaslt/.slurmlogs/%x-%j.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --nodes=1

# Activating conda

echo "Activating conda."
source /mnt/data/miniconda3/bin/activate

# Updating conda itself

conda update -n base -c defaults conda

# Upgrading to the libmamba solver

conda install -n base conda-libmamba-solver
conda config --set solver libmamba

# Creating conda environment if it does not already exists

env_yml_path=/mnt/slurm/lennart/jepaslt/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/mnt/data/miniconda3/envs/$env_name

conda env create --debug --file=$env_yml_path

# Activating the environment

conda activate $env_name

# Actually run the evaluation script

export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/jepaslt/
python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_0.yaml --devices cuda:0 cuda:1

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_1.yaml --devices cuda:0 cuda:1

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_2.yaml --devices cuda:0 cuda:1

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_3.yaml --devices cuda:0 cuda:1

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/jepaslt/configs/evals/vith16_384_ravdess_split_4.yaml --devices cuda:0 cuda:1

# I also need to "unexport" the path, because otherwise I will clutter this env variable.
export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/jepaslt/}
conda deactivate
conda env remove -n $env_name

conda config --set solver classic