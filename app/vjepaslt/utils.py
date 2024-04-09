import logging
import sys
from typing import Union
import torch
from src.models.utils.multimask import MultiMaskWrapper, SLTPredictorMultiMaskWrapper
import src.models.vision_transformer as vision_transformer
import src.models.slt_predictor as slt_predictor

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
    pred_depth: int = 6,
    pred_embed_dim: int = 384,
    uniform_power: bool = False,
    use_mask_tokens: bool = False,
    num_mask_tokens: int = 2,
    zero_init_mask_tokens: bool = True,
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
    encoder = MultiMaskWrapper(encoder)
    predictor = slt_predictor.__dict__['slt_predictor'](
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens
    )
    predictor = SLTPredictorMultiMaskWrapper(predictor)

    encoder.to(device)
    predictor.to(device)

    logger.info(encoder)
    logger.info(predictor)

    logger.info(
        f"Encoder number of trainable parameters: {count_trainable_parameters(encoder)}")
    logger.info(
        f"Encoder number of total parameters: {count_parameters(encoder)}")
    logger.info(
        f"Predictor number of trainable parameters: {count_trainable_parameters(predictor)}")
    logger.info(
        f"Predictor number of total parameters: {count_parameters(predictor)}")

    return encoder, predictor


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
