import argparse
import pathlib

import torch
import yaml
import logging


def get_argumentparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, type=pathlib.Path, help='config file')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help='log level')
    return parser

def init_encoder(
    device,
    load_path,
    model_name: str,
    patch_size: int = 16,
    crop_size: int = 224,
    frames_per_clip: int = 16,
    tubelet_size: int = 2,
    use_sdpa: bool = False,
    use_SiLU: bool = False,
    tight_SiLU: bool = True,
    uniform_power: bool = False,
    checkpoint_key: str = 'target_encoder'
):

    encoder = src.models.vision_transformer.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU
    ).to(device)

    checkpoint = torch.load(load_path, map_location='cpu')
    pretrained_dict = checkpoint[checkpoint_key]
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    
    encoder.load_state_dict(pretrained_dict)
    print(encoder)

    del checkpoint
    return encoder

def main(args):

    with open(args.file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # -- Setup logging

    logging.basicConfig(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # -- Load data config
    data_config = config['data']
    number_of_languages = data_config.get('number_of_languages', 1)

    # Load V-Jepa encoder as well as Target encoder

    pretrained_config = config['pretrained']
    encoder_config = pretrained_config['encoder']
    encoder = init_encoder('cuda:0', **encoder_config)

    # Load language specific models
    model_config = config['model']
    logger.info(model_config)

    # Load datasets



if __name__ == '__main__':
    parser = get_argumentparser()
    args = parser.parse_args()
    main(args)