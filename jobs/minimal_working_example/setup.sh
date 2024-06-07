# POTENTIAL SLURM COMMANDS
# TODO: Run this whole thing on a SLURM cluster and check if it works.

docker run --gpus all --ipc=host -v /home/lennarteing/jepaslt/:/workspace/jepaslt/ -v /etc/localtime:/etc/localtime:ro -it --rm jepaslt:latest jepaslt/jobs/minimal_working_example/runner.sh