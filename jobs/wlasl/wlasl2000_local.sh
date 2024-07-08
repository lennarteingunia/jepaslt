#!/bin/bash

source /home/lennarteing/miniforge3/bin/activate

env_yml_path=/mnt/slurm/lennart/jepaslt/jepaslt/environment.yml
env_name=$(grep 'name:' $env_yml_path | awk '{print $2}')
env_path=/home/lennarteing/miniforge3/envs/$env_name

if [ ! -d $env_path ]
then
	echo "Specified environment does not exist. CREATING..."
	conda env create --debug --file=$env_yml_path
else
	echo "Specified environment already exists. UPDATING..."
	conda env update -n $env_name --debug --file=$env_yml_path --prune
fi

conda activate $env_name

rsync -havzP --stats --delete /mnt/datasets/WLASL2000/WLASL2000_vjepa /mnt/data/
export PYTHONPATH=$PYTHONPATH:/mnt/slurm/lennart/jepaslt/jepaslt/
python /mnt/slurm/lennart/jepaslt/scripts/rewrite_csv.py --file=/mnt/data/WLASL2000_vjepa/train.csv --prefix=/mnt/data/WLASL2000_vjepa/train --output=/mnt/slurm/lennart/jepaslt/data/train.csv
python /mnt/slurm/lennart/jepaslt/scripts/rewrite_csv.py --file=/mnt/data/WLASL2000_vjepa/val.csv --prefix=/mnt/data/WLASL2000_vjepa/val --output=/mnt/slurm/lennart/jepaslt/data/val.csv

python -m evals.main --fname=/mnt/slurm/lennart/jepaslt/configs/evals/vith16_384_wlasl2000.yaml --devices cuda:0 cuda:1

# --- TEARDOWN ---

export PYTHONPATH=${PYTHONPATH%:/mnt/slurm/lennart/jepaslt/jepaslt/}
rm /mnt/slurm/lennart/jepaslt/data/train.csv
rm /mnt/slurm/lennart/jepaslt/data/val.csv