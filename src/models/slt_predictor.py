from functools import partial
from typing import Union
import torch

from src.models.utils.modules import MixedBlock


class MixedTransformerPredictor(torch.nn.Module):

    def __init__(
        self,
        *args,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        norm_layer=torch.nn.LayerNorm,
        **kwargs,
    ):
        super(MixedTransformerPredictor, self).__init__(*args, **kwargs)

        self._predictor_blocks = torch.nn.ModuleList([
            MixedBlock(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=torch.nn.GELU,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
            ) for _ in range(depth)
        ])

    def forward(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def mixed_predictor(**kwargs):
        model = MixedTransformerPredictor(
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            **kwargs
        )
        return model
