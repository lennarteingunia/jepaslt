#!/bin/bash

### Slurm Options ###
#SBATCH --job-name=jepaslt
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/logs/%x-%j.log
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
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

# Syncing dataset
echo "RSyncing datasets..."
rsync -havzP --stats --delete /mnt/datasets/WLASL2000/WLASL2000_vjepa /mnt/data/

# We need to rewrite the csvs, because they are not formatted correctly
# I need to specify the correct PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/jepaslt/
python /mnt/slurm/lennart/jepaslt/scripts/rewrite_csv.py --file=/mnt/data/WLASL2000_vjepa/train.csv --prefix=/mnt/data/WLASL2000_vjepa/train --output=/mnt/slurm/lennart/jepaslt/data/train.csv
python /mnt/slurm/lennart/jepaslt/scripts/rewrite_csv.py --file=/mnt/data/WLASL2000_vjepa/val.csv --prefix=/mnt/data/WLASL2000_vjepa/val --output=/mnt/slurm/lennart/jepaslt/data/val.csv

# Actually run the evaluation script

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_wlasl2000_minimal.yaml --devices cuda:0

# I also need to "unexport" the path, because otherwise I will clutter this env variable.
export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/jepaslt/}

# After this I also remove rewritten csvs in case anything changes in the future with the dataset
rm /mnt/slurm/lennart/jepaslt/data/train.csv
rm /mnt/slurm/lennart/jepaslt/data/val.csv
