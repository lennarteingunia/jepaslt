from functools import partial
import math
from typing import Union
import torch

from src.masks.utils import apply_masks
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.models.utils.modules import Block, CrossAttentionBlock
from src.utils.tensors import repeat_interleave_batch, trunc_normal_


class MultimodalVisionTransformerPredictor(torch.nn.Module):

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 1,
        tubelet_size: int = 2,
        embed_dim: int = 768,
        predictor_embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        norm_layer=torch.nn.LayerNorm,
        init_std: float = 0.02,
        uniform_power: bool = False,
        use_mask_tokens: bool = False,
        num_mask_tokens: int = 2,
        zero_init_mask_tokens: bool = True,
    ):
        super(MultimodalVisionTransformerPredictor, self).__init__()

        self._predictor_embed_dim = predictor_embed_dim
        self._predictor_embed = torch.nn.Linear(
            embed_dim,
            predictor_embed_dim,
            bias=True
        )

        self._mask_tokens = None
        self._num_mask_tokens = 0
        if use_mask_tokens:
            self._num_mask_tokens = num_mask_tokens
            self._mask_tokens = torch.nn.ParameterList([
                torch.nn.Parameter(torch.zeros(1, 1, predictor_embed_dim)) for _ in range(num_mask_tokens)
            ])

        self._input_size = img_size
        self._patch_size = patch_size

        self._num_frames = num_frames
        self._tubelet_size = tubelet_size
        self._is_video = num_frames > 1

        self._grid_size = self._input_size // self._patch_size
        self._grid_depth = self._num_frames // self._tubelet_size

        if self._is_video:
            self._num_patches = num_patches = (
                num_frames // tubelet_size) * (img_size // patch_size) * (img_size // patch_size)
        else:
            self._num_patches = num_patches = (
                img_size // patch_size) * (img_size // patch_size)

        self._uniform_power = uniform_power
        self._predictor_pos_embed = torch.nn.Parameter(
            torch.zeros(1, num_patches, predictor_embed_dim),
            requires_grad=False
        )

        self._predictor_blocks = torch.nn.ModuleList([
            Block(
                dim=predictor_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer
            ) for _ in range(depth)
        ])

        self._predictor_norm = norm_layer(predictor_embed_dim)
        self._predictor_proj = torch.nn.Linear(
            predictor_embed_dim,
            embed_dim,
            bias=True
        )

        if not self._predictor_pos_embed is None:
            self._init_pos_embed(self._predictor_pos_embed.data)
        self._init_std = init_std
        if not zero_init_mask_tokens:
            for mask_token in self._mask_tokens:
                trunc_normal_(mask_token, std=init_std)
        self.apply(self._init_weights)

        # TODO: What is the layer rescaling for in the original?

    def _init_pos_embed(self, pos_embed):
        if self._is_video:
            sincos = get_3d_sincos_pos_embed(
                self._predictor_embed_dim,
                self._grid_size,
                self._grid_depth,
                cls_token=False,
                uniform_power=self._uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(
                self._predictor_embed_dim,
                self._grid_size,
                cls_token=False
            )
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            trunc_normal_(m.weight, std=self._init_std)
            if not m.bias is None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _diffusion(self, x, noise_beta=(0.5, 1.0), steps=1000):
        # Prepare diffusion noise schedule
        b1, b2 = noise_beta
        beta_scheduler = (b1 + i*(b2-b1)/steps for i in range(steps))
        alpha_scheduler = []
        _alpha = 1.0
        for _beta in beta_scheduler:
            _alpha *= 1.-_beta
            alpha_scheduler += [_alpha]

        # Sample diffusion time step
        T = torch.randint(0, steps, (len(x),))
        alpha = torch.tensor(alpha_scheduler, device=x.device)[
            T].unsqueeze(-1).unsqueeze(-1)

        # Normalize features and apply noise
        x = torch.nn.functional.layer_norm(x, (x.size(-1),))
        x = alpha**0.5 * x + (1.-alpha)**0.5 * \
            torch.randn(x.shape, device=x.device)
        return x

    def forward(
        self,
        img_ctxt: torch.Tensor,
        lang_ctxt: torch.Tensor,
        mask_tgt: torch.Tensor,
        masks_ctxt_indices: torch.Tensor,
        masks_tgt_indices: torch.Tensor,
        mask_index: int = 1
    ) -> torch.Tensor:

        if not isinstance(masks_ctxt_indices, list):
            masks_ctxt_indices = [masks_ctxt_indices]

        if not isinstance(masks_tgt_indices, list):
            masks_tgt_indices = [masks_tgt_indices]

        B = len(img_ctxt) // len(masks_ctxt_indices)

        x_1 = self._predictor_embed(img_ctxt)
        x_2 = self._predictor_embed(lang_ctxt)
        _, N_ctxt, _ = x_1.shape
        _, N_lang_ctxt, _ = lang_ctxt.shape

        if self._predictor_pos_embed:
            img_ctxt_pos_embed = self._predictor_pos_embed.repeat(B, 1, 1)
            x_1 += apply_masks(img_ctxt_pos_embed, masks_ctxt_indices)

        if not self._mask_tokens:
            pred_tokens = self._predictor_embed(mask_tgt)
            pred_tokens = self._diffusion(pred_tokens)
        else:
            mask_index = mask_index % self._num_mask_tokens
            pred_tokens = self._mask_tokens[mask_index]
            pred_tokens = pred_tokens.repeat(B, self._num_patches, 1)
            pred_tokens = apply_masks(pred_tokens, masks_tgt_indices)

        if self._predictor_pos_embed:
            pos_embeds = self._predictor_pos_embed.repeat(B, 1, 1)
            pos_embeds = apply_masks(pos_embeds, masks_tgt_indices)
            pos_embeds = repeat_interleave_batch(
                pos_embeds, B, repeat=len(masks_ctxt_indices))

        x_1 = x_1.repeat(len(masks_tgt_indices), 1, 1)
        x = torch.cat([x_2, x_1, pred_tokens], dim=1)

        masks_ctxt_indices = torch.cat(masks_ctxt_indices, dim=0)
        masks_tgt_indices = torch.cat(masks_tgt_indices, dim=0)
        masks = torch.cat([masks_ctxt_indices, masks_tgt_indices], dim=1)

        for block in self._predictor_blocks:
            # TODO Prepend lang_ctxt to image tokens
            x = block(x, masks)
        x = self._predictor_norm(x)

        x = x[:, N_ctxt:]
        x = self._predictor_proj(x)

        return x


def slt_predictor(**kwargs):
    model = MultimodalVisionTransformerPredictor(
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        **kwargs
    )
    return model

if __name__ == "__main__":
    predictor = M
