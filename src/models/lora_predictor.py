from __future__ import annotations
from functools import partial
from typing import Tuple, Union

import loralib
import torch.nn as nn
import src.models.predictor as predictor

from src.models.utils.modules import MLP, Attention


class LoRAVisionTransformerPredictor(predictor.VisionTransformerPredictor):

    def __init__(
        self: LoRAVisionTransformerPredictor,
        rank: int = 8,
        lora_qkv_enabled: Tuple[bool, bool, bool] = (True, False, True),
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Union[None, float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer=nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        use_mask_tokens: bool = False,
        num_mask_tokens: int = 2,
        zero_init_mask_tokens: bool = True,
        **kwargs
    ) -> None:
        predictor.VisionTransformerPredictor.__init__(
            self=self,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            **kwargs
        )
        self.predictor_embed = loralib.Linear(
            in_features=self.predictor_embed.in_features,
            out_features=self.predictor_embed.out_features,
            r=rank,
            bias=True,
            **kwargs
        )
        self.predictor_proj = loralib.Linear(
            in_features=self.predictor_proj.in_features,
            out_features=self.predictor_proj.out_features,
            r=rank,
            bias=True,
            **kwargs
        )
        self.apply(partial(self._init_lora_modules, rank=rank,
                   lora_qkv_enabled=lora_qkv_enabled, qkv_bias=qkv_bias, **kwargs))

    def _init_lora_modules(
        self: predictor.VisionTransformerPredictor,
        module: nn.Module,
        *,
        rank: int,
        lora_qkv_enabled: Tuple[bool, bool, bool],
        qkv_bias: bool,
        **kwargs
    ) -> None:
        if isinstance(module, Attention):
            module.proj = loralib.Linear(
                in_features=module.proj.in_features,
                out_features=module.proj.out_features,
                r=rank,
                **kwargs
            )
            module.qkv = loralib.MergedLinear(
                in_features=module.qkv.in_features,
                out_features=module.qkv.out_features,
                r=rank,
                enable_lora=lora_qkv_enabled,
                bias=qkv_bias,
                **kwargs
            )
        if isinstance(module, MLP):
            module.fc1 = loralib.Linear(
                in_features=module.fc1.in_features,
                out_features=module.fc1.out_features,
                r=rank,
                **kwargs
            )
            module.fc2 = loralib.Linear(
                in_features=module.fc2.in_features,
                out_features=module.fc2.out_features,
                r=rank,
                **kwargs
            )

    def from_vit_predictor(
        vit_predictor: predictor.VisionTransformerPredictor,
        rank: int = 8,
        lora_qkv_enabled: Tuple[bool, bool, bool] = (True, False, True),
        zero_init_mask_tokens: bool = True,
        **kwargs
    ) -> LoRAVisionTransformerPredictor:
        model = LoRAVisionTransformerPredictor(
            rank=rank,
            lora_qkv_enabled=lora_qkv_enabled,
            img_size=vit_predictor.input_size,
            patch_size=vit_predictor.patch_size,
            num_frames=vit_predictor.num_frames,
            tubelet_size=vit_predictor.tubelet_size,
            embed_dim=vit_predictor.predictor_embed.in_features,
            predictor_embed_dim=vit_predictor.predictor_embed.out_features,
            depth=len(vit_predictor.predictor_blocks),
            num_heads=vit_predictor.predictor_blocks[0].attn.num_heads,
            mlp_ratio=(
                vit_predictor.predictor_blocks[0].mlp.fc1.out_features // vit_predictor.predictor_embed.out_features),
            qkv_bias=vit_predictor.predictor_blocks[0].attn.qkv.bias is not None,
            qk_scale=vit_predictor.predictor_blocks[0].attn.scale,
            drop_rate=vit_predictor.predictor_blocks[0].attn.proj_drop_prob,
            attn_drop_rate=vit_predictor.predictor_blocks[0].attn.attn_drop.p,
            norm_layer=type(vit_predictor.predictor_norm),
            init_std=vit_predictor.init_std,
            uniform_power=vit_predictor.uniform_power,
            use_mask_tokens=vit_predictor.num_mask_tokens != 0,
            num_mask_tokens=vit_predictor.num_mask_tokens,
            zero_init_mask_tokens=zero_init_mask_tokens,
            **kwargs
        )
        model.load_state_dict(vit_predictor.state_dict(), strict=False)
        return model
