#!/bin/bash

python -m app.vjepaslt.train fit -c $PWD/configs/pretrain/minimal_working_example.yaml --trainer.logger.init_args.save_dir $PWD/.logs/$(date +%Y%m%d%H%M%S) --trainer.logger.init_args.project BIGEKO