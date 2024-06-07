#!/bin/bash

cd jepaslt
python -m app.vjepaslt.train --config configs/pretrain/minimal_working_example.yaml --accelerator cuda --devices 0 1 --num-nodes 1 --strategy=ddp --root .logs/