import argparse
import logging
import os
import shutil
import sys
import time
from typing import Optional, Union
import lightning
import yaml

def build_lightning_module(vjepa_cfg: dict, model_cfg: dict):
    pass

def get_logger(
    *,
    name: Optional[str] = None,
    level: Union[str, int] = logging.INFO,
    format: str = "[%(levelname)-8s][%(asctime)s][%(funcName)-25s] %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    logging.basicConfig(
        stream=sys.stdout,
        level=level,
        format=format,
        datefmt=datefmt
    )
    return logging.getLogger(name=name)

def get_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=str, help='path to config file')
    parser.add_argument('--root', required=True, type=str, help='path to root  for logging etc.')
    parser.add_argument('--accelerator', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--devices', type=int, nargs='+', default=[0])
    parser.add_argument('--num-nodes', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='ddp')
    parser.add_argument('--test', action='store_true')
    return parser

def cleanup(
    logger: logging.Logger,
    logging_root: str,
    *,
    test_mode: Optional[bool] = None   
) -> None:
    logger.info(f'Performing cleanup routine.')
    if test_mode:
        logger.info(f'Performing test mode cleanup routine.')
        logger.debug(f'Removing {logging_root=}')
        shutil.rmtree(logging_root)
    

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    return cfg

def main(program_args: argparse.Namespace, cfg: dict) -> None:

    logging_cfg = cfg.get('logging', dict())
    logger = get_logger(**logging_cfg)
    if len(logging_cfg) == 0:
        logger.info('No logging configuration was given, using default.')

    logging_root = os.path.join(program_args.root, cfg['experiment'], time.strftime("%Y%m%d-%H%M%S"))
    logger.info(f'Creating logging folder at {logging_root}')
    os.makedirs(logging_root, exist_ok=True)
    logger.info(f'Copying used configuration to {logging_root}.')
    shutil.copy2(program_args.config, logging_root)

    logger.info(f'Creating lightning trainer using\n\t{logging_root=}\n\t{program_args.accelerator=}\n\t{program_args.devices=}\n\t{program_args.num_nodes=}\n\t{program_args.strategy=}')

    trainer = lightning.Trainer(
        default_root_dir=logging_root,
        accelerator=program_args.accelerator,
        devices=program_args.devices,
        num_nodes=program_args.num_nodes,
        strategy=program_args.strategy
    )

    cleanup(
        logger=logger,
        logging_root=logging_root, 
        test_mode=program_args.test
    )
    

if __name__ == "__main__":
    parser = get_argument_parser()
    program_args = parser.parse_args()
    cfg = load_config(program_args.config)
    main(program_args, cfg)
