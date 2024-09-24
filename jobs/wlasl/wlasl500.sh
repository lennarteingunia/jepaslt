#!/bin/bash

### Slurm Options ###
#SBATCH --job-name=jepaslt-wlasl500
#SBATCH --time=24-0
#SBATCH --output=/mnt/slurm/lennart/jepaslt/logs/%x-%j.log
#SBATCH --cpus-per-task=10
#SBATCH --mem=40G
#SBATCH --nodes=1

# Activating conda

source /mnt/data/miniconda3/bin/activate

# Creating conda environment if it does not already exists

env_yml_path=/mnt/slurm/lennart/jepaslt/environment.yml
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
rsync -havzP --stats --delete /mnt/datasets/wlasl/wlasl_vjepa /mnt/data/

# We need to rewrite the csvs, because they are not formatted correctly
# I need to specify the correct PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/jepaslt/
python /mnt/slurm/lennart/jepaslt/slurm/rewrite_filepath_prefixes_in_csv.py --file=/mnt/data/wlasl_vjepa/train500.csv --prefix=/mnt/data/wlasl_vjepa/train --output=/mnt/slurm/lennart/jepaslt/data/train500.csv
python /mnt/slurm/lennart/jepaslt/slurm/rewrite_filepath_prefixes_in_csv.py --file=/mnt/data/wlasl_vjepa/val500.csv --prefix=/mnt/data/wlasl_vjepa/val --output=/mnt/slurm/lennart/jepaslt/data/val500.csv

# Actually run the evaluation script

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_wlasl500_16x8x3.yaml --devices cuda:0 cuda:1 cuda:2

# I also need to "unexport" the path, because otherwise I will clutter this env variable.
export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/jepaslt/}

# After this I also remove rewritten csvs in case anything changes in the future with the dataset
rm /mnt/slurm/lennart/jepaslt/data/train500.csv
rm /mnt/slurm/lennart/jepaslt/data/val500.csv
