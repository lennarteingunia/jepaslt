import logging
import torch

import src.models.vision_transformer

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_predictor(
    *,
    device: torch.device,
    pretrained_checkpoint_path: str,
    patch_size: int = 16,
    frames_per_clip: int = 16,
    tubelet_size: int = 2,
    crop_size: int = 224,
    depth: int = 6,
    num_heads: int,
    encoder_embed_dim: int,
    embed_dim: int = 384,
    uniform_power: bool = False,
    use_mask_tokens: bool = False,
    num_mask_tokens: int = 2,
    zero_init_mask_tokens: bool = True,
    use_sdpa: bool = False,
) -> torch.nn.Module:
    predictor = src.models.vision_transformer.__dict__['vit_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        embed_dim=encoder_embed_dim,
        predictor_embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa
    )

    predictor = predictor.to(device)
    predictor = load_predictor(
        predictor=predictor, 
        pretrained_checkpoint_path=pretrained_checkpoint_path, 
        checkpoint_key='predictor'
    )
    return predictor

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

def load_predictor(
    predictor: torch.nn.Module,
    pretrained_checkpoint_path: str,
    checkpoint_key: str
) -> torch.nn.Module:
    logger.info(f"Loading pretrained predictor model {pretrained_checkpoint_path}")
    checkpoint = torch.load(pretrained_checkpoint_path, map_location="cpu")
    pretrained_dict = checkpoint[checkpoint_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = predictor.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded pretrained predictor with message: {msg}")
    logger.info(f"Loaded pretrained predictor from epoch: {checkpoint['epoch']}\npatch:{pretrained_checkpoint_path}")
    del checkpoint
    return predictor

def load_encoder(
    encoder: torch.nn.Module,
    pretrained_checkpoint_path: str,
    checkpoint_key: str
) -> torch.nn.Module:
    logger.info(f"Loading pretrained encoder model from {pretrained_checkpoint_path}")
    checkpoint = torch.load(pretrained_checkpoint_path, map_location="cpu")
    pretrained_dict = checkpoint[checkpoint_key]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Loaded pretrained encoder with message: {msg}")
    logger.info(f"Loaded pretrained encoder from epoch: {checkpoint['epoch']}\npatch:{pretrained_checkpoint_path}")
    del checkpoint
    return encoder
