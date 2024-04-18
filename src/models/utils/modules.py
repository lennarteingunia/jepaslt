# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
from __future__ import annotations
from typing import List, Tuple, Union

import loralib
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LoRAMLP(nn.Module):

    def __init__(
        self: LoRAMLP,
        rank: Union[Tuple[int, int], int],
        in_features: int,
        hidden_features: Union[None, int] = None,
        out_features: Union[None, int] = None,
        act_layer=nn.GELU,
        drop: float = 0.,
        **additional_lora_params,
    ) -> None:
        super(LoRAMLP, self).__init__()
        rank = rank if isinstance(rank, tuple) else rank, rank
        assert len(rank) == 2, "Too many ranks were given for a two layer MLP."
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self._fc1 = loralib.Linear(
            in_features, out_features=hidden_features, rank=rank[0], **additional_lora_params)
        self._act = act_layer()
        self._fc2 = loralib.Linear(
            hidden_features, out_features=out_features, rank=rank[1], **additional_lora_params)
        self._drop = nn.Dropout(drop)

    def forward(
        self: LoRAMLP,
        x: torch.Tensor
    ) -> torch.Tensor:
        x = self._fc1(x)
        x = self._act(x)
        x = self._drop(x)
        x = self._fc2(x)
        x = self._drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, x, _mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, N, D]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * \
                self.scale  # [B, num_heads, D, D]
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LoRAAttention(nn.Module):

    def __init__(
        self: LoRAAttention,
        rank: int,
        dim: int,
        qkv_lora_enable: List[bool] = [True, False, True],
        use_lora_for_inner_projection: bool = False,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        **additional_lora_params,
    ) -> None:
        super(LoRAAttention, self).__init__()
        assert len(
            qkv_lora_enable) == 3, f"{LoRAAttention.__name__} needs 3 truth values wether or not LoRA should be enabled."
        self._num_heads = num_heads
        head_dim = dim // num_heads
        self._scale = qk_scale or head_dim ** -.5
        self._qkv = loralib.MergedLinear(
            in_features=dim,
            out_features=dim * 3,
            r=rank,
            enable_lora=qkv_lora_enable,
            bias=qkv_bias
            ** additional_lora_params
        )
        self._attn_drop = nn.Dropout(attn_drop)

        self._proj = loralib.Linear(
            in_features=dim,
            out_features=dim,
            r=rank,
            **additional_lora_params
        ) if use_lora_for_inner_projection else nn.Linear(
            in_features=dim,
            out_features=dim
        )

        self._proj_drop_prob = proj_drop
        self._proj_drop = nn.Dropout(proj_drop)

    def forward(
        self: LoRAAttention,
        x: torch.Tensor,
        _mask: Union[None, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self._qkv(x).reshape(B, N, 3, self._num_heads,
                                   C // self._num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv
        attn = (q @ k.transpose(-2, -1)) * self._scale
        attn = attn.softmax(dim=-1)
        attn = self._attn_drop(attn)
        x = (attn @ v)
        x.transpose(1, 2).reshape(B, N, C)
        x = self._proj(x)
        x = self._proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        grid_size=None,
        grid_depth=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x, return_attention=False, mask=None):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class LoRABlock(nn.Module):

    def __init__(
        self: LoRABlock,
        rank: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **additional_lora_params
    ) -> None:
        super(LoRABlock, self).__init__()

        self._norm1 = norm_layer(dim)
        self._attn = LoRAAttention(
            rank=rank,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **additional_lora_params
        )
        self._norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self._mlp = LoRAMLP(
            rank=rank,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
            ** additional_lora_params
        )

    def forward(
        self: LoRABlock,
        x: torch.Tensor,
        return_attention: bool = False,
        mask: Union[None, torch.Tensor] = None,
    ) -> torch.Tensor:
        y, attn = self._attn(self._norm1(x), _mask=mask)
        if return_attention:
            return attn
        x = x + y
        x = x + self._mlp(self._norm2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.,
        proj_drop=0.,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop_prob = proj_drop
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_sdpa = use_sdpa

    def forward(self, q, x, _mask: Union[None, torch.Tensor] = None):
        B, N_1, C = q.shape
        q = self.q(q).reshape(B, N_1, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        B, N_2, C = x.shape
        kv = self.kv(x).reshape(B, N_2, 2, self.num_heads, C //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        # (batch_size, num_heads, seq_len, feature_dim_per_head)
        k, v = kv[0], kv[1]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self.proj_drop_prob)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # (batch_size, num_heads, query_len, seq_len)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            q = (attn @ v)

        q = q.transpose(1, 2).reshape(B, N_1, C)
        q = self.proj(q)
        q = self.proj_drop(q)
        return q, attn


class LoRACrossAttention(nn.Module):

    def __init__(
        self: LoRACrossAttention,
        rank: int,
        dim: int,
        enable_lora: List[bool] = [True, False, True],
        use_lora_for_inner_proj: bool = False,
        num_heads: int = 12,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        **additional_lora_params,
    ) -> None:
        super(LoRACrossAttention, self).__init__()
        assert len(
            enable_lora) == 3, f"{LoRACrossAttention.__name__} needs you to give 3 truth values for wether or not to use LoRA for Q, K, V."
        self._num_heads = num_heads
        head_dim = dim // num_heads
        self._scale = qk_scale or head_dim ** -.5

        self._q = loralib.Linear(
            in_features=dim,
            out_features=dim,
            r=rank,
            bias=qkv_bias,
            **additional_lora_params
        ) if enable_lora[0] else nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=qkv_bias
        )

        self._k = loralib.Linear(
            in_features=dim,
            out_features=dim,
            r=rank,
            bias=qkv_bias,
            **additional_lora_params
        ) if enable_lora[1] else nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=qkv_bias
        )

        self._v = loralib.Linear(
            in_features=dim,
            out_features=dim,
            r=rank,
            bias=qkv_bias,
            **additional_lora_params
        ) if enable_lora[2] else nn.Linear(
            in_features=dim,
            out_features=dim,
            bias=qkv_bias
        )

        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = loralib.Linear(
            in_features=dim,
            out_features=dim,
            r=rank,
            **additional_lora_params
        ) if use_lora_for_inner_proj else nn.Linear(
            in_features=dim,
            out_features=dim
        )

        self._proj_drop_prob = proj_drop
        self._proj_drop = nn.Dropout(proj_drop)

    def forward(
        self: LoRACrossAttention,
        q: torch.Tensor,
        x: torch.Tensor,
        _mask: Union[None, torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N_1, C = q.shape
        q = self._q(q).reshape(B, N_1, self._num_heads, C //
                               self._num_heads).permute(0, 2, 1, 3)

        B, N_2, C = x.shape
        k = self._k(x).reshape(B, N_2, self._num_heads, C //
                               self._num_heads).permute(0, 2, 1, 3)
        v = self._v(x).reshape(B, N_2, self._num_heads, C //
                               self._num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self._scale
        attn = attn.softmax(dim=-1)
        attn = self._attn_drop(attn)
        q = (attn @ v)
        q = q.transpose(1, 2).reshape(B, N_1, C)
        q = self._proj(q)
        q = self._proj_drop(q)
        return q, attn


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(self, q, x, return_attention: bool = False, mask: Union[None, torch.Tensor] = None):
        y, attn = self.xattn(q, self.norm1(x), mask=mask)
        if return_attention:
            return attn
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q


class LoRACrossAttentionBlock(nn.Module):

    def __init__(
        self: LoRACrossAttentionBlock,
        rank: int,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Union[None, float] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **additional_lora_params,
    ) -> None:
        super(LoRACrossAttentionBlock, self).__init__()
        self._norm1 = norm_layer(dim)
        self._xattn = LoRACrossAttention(
            rank=rank,
            dim=dim,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            **additional_lora_params
        )
        self._norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self._mlp = LoRAMLP(
            rank=rank,
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            **additional_lora_params
        )

    def forward(
        self: LoRACrossAttentionBlock,
        q: torch.Tensor,
        x: torch.Tensor,
        return_attention: bool = False,
        mask: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        y, attn = self._xattn(q, self._norm1(x), _mask=mask)
        if return_attention:
            return attn
        q = q + y
        q = q + self._mlp(self._norm2(q))
        return q
