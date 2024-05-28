from __future__ import annotations
from functools import partial
from typing import Tuple

import torch.nn as nn
import loralib
import src.models.vision_transformer as vision_transformer

from src.models.utils.modules import MLP, Attention


class LoRAVisionTransformer(vision_transformer.VisionTransformer):

    def __init__(
        self: LoRAVisionTransformer,
        rank: int = 8,
        lora_qkv_enabled: Tuple[bool, bool, bool] = (True, False, True),
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
        uniform_power=False,
        **kwargs
    ) -> None:
        vision_transformer.VisionTransformer.__init__(
            self,
            img_size=img_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_std=init_std,
            out_layers=out_layers,
            uniform_power=uniform_power,
            **kwargs
        )
        self.apply(partial(self._init_lora_modules, rank=rank,
                   lora_qkv_enabled=lora_qkv_enabled, qkv_bias=qkv_bias, **kwargs))

    def _init_lora_modules(
        self: LoRAVisionTransformer,
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

    def from_vit(
        vit: vision_transformer.VisionTransformer,
        rank: int = 8,
        lora_qkv_enabled: Tuple[bool, bool, bool] = (True, False, True),
        **kwargs
    ) -> LoRAVisionTransformer:
        model = LoRAVisionTransformer(
            rank=rank,
            lora_qkv_enabled=lora_qkv_enabled,
            img_size=vit.input_size,
            patch_size=vit.patch_size,
            num_frames=vit.num_frames,
            tubelet_size=vit.tubelet_size,
            in_chans=vit.patch_embed.proj.in_channels,
            embed_dim=vit.embed_dim,
            depth=len(vit.blocks),
            mlp_ratio=(vit.blocks[0].mlp.fc1.out_features // vit.embed_dim),
            qkv_bias=vit.blocks[0].attn.qkv.bias is not None,
            qk_scale=vit.blocks[0].attn.scale,
            drop_rate=vit.blocks[0].attn.proj_drop_prob,
            attn_drop_rate=vit.blocks[0].attn.attn_drop.p,
            norm_layer=type(vit.norm),
            init_std=vit.init_std,
            out_layers=vit.out_layers,
            uniform_power=vit.uniform_power,
            **kwargs
        )
        model.load_state_dict(vit.state_dict(), strict=False)
        return model