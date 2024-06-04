import logging
import torch

import src.models.vision_transformer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_encoder(
    *,
    device: torch.device,
    pretrained_checkpoint_path: str,
    model_name: str,
    patch_size: int = 16,
    crop_size: int = 224,
    frames_per_clip: int = 16,
    tubelet_size: int = 2,
    use_sdpa: bool = False,
    use_SiLU: bool = False,
    tight_SiLU: bool = True,
    uniform_power: bool = False,
    checkpoint_key: str = "target_encoder"
) -> torch.nn.Module:
    
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

    encoder = load_encoder(
        encoder=encoder, 
        pretrained_checkpoint_path=pretrained_checkpoint_path, 
        checkpoint_key=checkpoint_key
    )

    return encoder


def load_encoder(
    encoder: torch.nn.Module,
    pretrained_checkpoint_path: str,
    checkpoint_key: str
) -> torch.nn.Module:
    logger.info(f"Loading pretrained model from {pretrained_checkpoint_path}")
    checkpoint = torch.load(pretrained_checkpoint_path, map_location="cpu")
    pretrained_dict = checkpoint[checkpoint_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", "")                       : v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded pretrained encoder with message: {msg}")
    logger.info(f"Loaded pretrained encoder fom epoch: {checkpoint['epoch']}\npatch:{pretrained_checkpoint_path}")
    del checkpoint
    return encoder
