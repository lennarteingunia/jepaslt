import json
import logging
import os
import pathlib
import sys
import definitions
from typing import Tuple, Union
import torch
from src.models.utils.multimask import MultiMaskWrapper, SLTPredictorMultiMaskWrapper
import src.models.vision_transformer as vision_transformer
import src.models.slt_predictor as slt_predictor
from src.utils.schedulers import CosineWDSchedule, WarmupCosineSchedule

import torch.nn.functional

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


def init_opt(
    encoder: torch.nn.Module,
    predictor: torch.nn.Module,
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: int,
    num_epochs: int,
    weight_decay: float = 1e-6,
    final_weight_decay: float = 1e-6,
    final_lr: float = 0.0,
    mixed_precision: bool = False,
    ipe_scale: float = 1.25,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    zero_init_bias_weight_decay: bool = True
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if "bias" not in n and len(p.shape) != 1)
        },
        {
            "params": (p for n, p in predictor.named_parameters() if "bias" not in n and len(p.shape) != 1)
        },
        {
            "params": (p for n, p in encoder.named_parameters() if "bias" in n or len(p.shape) == 1),
            "WD_exclude": zero_init_bias_weight_decay,
            "weight_decay": 0
        },
        {
            "params": (p for n, p in predictor.named_parameters() if "bias" in n or len(p.shape) == 1),
            "WD_exclude": zero_init_bias_weight_decay,
            "weight_decay": 0,
        }
    ]

    logger.info("Using AdamW optimizer.")
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )
    weight_decay_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_weight_decay,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch)
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, weight_decay_scheduler


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())