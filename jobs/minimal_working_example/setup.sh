# POTENTIAL SLURM COMMANDS

docker run --gpus all --ipc=host -v /home/lennarteing/jepaslt/:/workspace/jepaslt/ -v /etc/localtime:/etc/localtime:ro -it --rm jepaslt:latest jepaslt/jobs/minimal_working_example/runner.sh