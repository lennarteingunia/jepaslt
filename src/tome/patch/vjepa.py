from typing import Optional
import warnings
import torch

from models.utils.modules import Attention, Block
from tome.merge import bipartite_soft_matching, merge_source, merge_wavg
from tome.utils import parse_r


class ToMeAttention(Attention):

    def forward(
        self,
        x: torch.Tensor,
        *,
        _mask: Optional[torch.Tensor] = None,
        size: Optional[torch.Tensor] = None
    ):
        # TODO: Output typing
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                      self.num_heads).permute(2, 0, 3, 1, 4)
        if self.use_sdpa:
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
                x = torch.nn.functional.scaled_dot_product_attention(
                    query=q,
                    key=k,
                    value=v,
                    dropout_p=self.proj_drop_prob
                )
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale

            if size is not None:

                attn = attn + size.log()[:, None, None, :, 0]

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn, k.mean()


def make_block_class(block_class):

    class ToMeBlock(block_class):

        def forward(
            self,
            x: torch.Tensor,
            return_attention: bool = False,
            mask: Optional[torch.Tensor] = None
        ) -> None:

            attn_size = self._tome_info['size'] if self._tome_info['prop_attn'] else None

            y, attn, metric = self.attn(self.norm1(x), size=attn_size)

            if return_attention:
                return attn

            x = x + y

            r = self._tome_info['r'].pop(0)
            if r > 0:

                merge, _ = bipartite_soft_matching(
                    metric=metric,
                    r=r,
                    class_token=self._tome_info['class_token'],
                    distill_token=self._tome_info['distill_token'],
                )

                if self._tome_info['trace_source']:
                    self._tome_info['source'] = merge_source(
                        merge=merge,
                        x=x,
                        source=self._tome_info['source']
                    )

                x, self._tome_info['size'] = merge_wavg(
                    merge=merge,
                    x=x,
                    size=self._tome_info['size']
                )

            return x + self.mlp(self.norm2(x))

    return ToMeBlock


def make_vision_transformer_class(transformer_class):

    class ToMeVisionTransformer(transformer_class):

        def forward(self, *args, **kwargs) -> torch.Tensor:
            self._tome_info['r'] = parse_r(len(self.blocks), self.r)
            self._tome_info['size'] = None
            self._tome_info['source'] = None
            return super(ToMeVisionTransformer, self).forward(*args, **kwargs)

    return ToMeVisionTransformer


def apply_patch(model, trace_source: bool = False, prop_attn: bool = False):

    if model.__class__.__name__ == 'ToMeVisionTransformer':
        warnings.warn(
            f'Not patching the given model, since it has already been patched previously.')
        return

    BlockClass = None
    TransformerClass = model.__class__

    for module in model.modules():
        if module.__class__.__name__ == 'Block':
            BlockClass = module.__class__

    if BlockClass is None:
        warnings.warn(
            f'Error patching model of type {model.__class__.__name__}. It is not a Vision Transformer')

    ToMeBlock = make_block_class(BlockClass)
    ToMeVisionTransformer = make_vision_transformer_class(TransformerClass)

    model.__class__ = ToMeVisionTransformer
    model.r = 0
    model._tome_info = {
        'r': model.r,
        'size': None,
        'source': None,
        'trace_source': trace_source,
        'prop_attn': prop_attn,
        'class_token': False,
        'distill_token': False,
    }

    for module in model.modules():
        if isinstance(module, BlockClass):
            module.__class__ = ToMeBlock
            module._tome_info = model._tome_info
        elif isinstance(module, Attention):
            module.__class__ = ToMeAttention
