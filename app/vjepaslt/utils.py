import logging
import sys
from typing import Union
import torch
import src.models.vision_transformer as vision_transformer

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def init_video_model(
    *,
    device: Union[torch.cuda.device, str],
    patch_size: int = 16,
    num_frames: int = 16,
    tubelet_size: int = 2,
    model_name: str = "vit_base",
    crop_size: int = 224,
    uniform_power: bool = False,
    use_sdpa: bool = False,
):
    encoder = vision_transformer.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
    )

    encoder.to(device)

    logger.info(encoder)
    logger.info(
        f"Encoder number of trainable parameters: {count_trainable_parameters(encoder)}")
    logger.info(
        f"Encoder number of total parameters: {count_parameters(encoder)}")

    return encoder


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
