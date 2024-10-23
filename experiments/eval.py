import argparse
import os
from typing import Any, Dict
import torch
import torch.amp
import torch.multiprocessing.spawn
import torch.multiprocessing.spawn
import yaml

from experiments import build_encoder, load_config


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '37137'
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(
        backend='nccl', rank=rank, world_size=world_size)


def run_eval(
    rank: int,
    world_size: int,
    config: Dict[str, Any],
):
    ddp_setup(rank, world_size)

    encoder = build_encoder(config=config)

    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Run evaluation using V-JEPA.')
    parser.add_argument('--config-path', type=str, required=True)
    args = parser.parse_args()

    config = load_config(config_path=args.config_path)

    world_size = torch.cuda.device_count()
    max_rank = world_size
    torch.multiprocessing.spawn(run_eval, args=(world_size, config,), nprocs=max_rank)
