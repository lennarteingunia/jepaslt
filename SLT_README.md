## How to run:

First you need to build the docker image:
```
docker build --pull --rm -f "Dockerfile" -t jepaslt:latest "."
```
After building the docker image, you should be able to simply run the code by choosing any `setup.sh` scripts from `jobs/` and running it.
```
./jobs/<your-experiment>/setup.sh
```
This will mount the repository into the correct docker container and run the specified experiment.