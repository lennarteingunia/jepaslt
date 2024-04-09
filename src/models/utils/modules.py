# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Union
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

    def forward(self, x, mask=None):
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


class MixedAttention(nn.Module):
    def __init__(
            self,
            *args,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_scale: Union[float, None] = None,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            use_sdpa: bool = True,
            **kwargs,
    ) -> None:
        super(MixedAttention, self).__init__(*args, **kwargs)

        self._num_heads = num_heads
        head_dim = dim // num_heads
        self._scale = qk_scale or head_dim ** -0.5
        self._qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self._q = nn.Linear(dim, dim, bias=qkv_bias)
        self._kv = nn.Linear(dim, dim, bias=qkv_bias)
        self._attn_drop = nn.Dropout(attn_drop)
        self._proj = nn.Linear(dim, dim)
        self._proj_drop_prob = proj_drop
        self._proj_drop = nn.Dropout(proj_drop)
        self._use_sdpa = use_sdpa

    def forward(
        self,
        image_features: torch.Tensor,
        language_features:
        torch.Tensor,
        mask: Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        B, I, C = image_features.shape
        _, L, _ = language_features.shape

        q = self._q(image_features).reshape(B, I, 1, self._num_heads,
                                            C // self._num_heads).permute(2, 0, 3, 1, 4)
        kv = self._kv(language_features).reshape(
            B, L, 2, self._num_heads, C // self._num_heads).permute(2, 0, 3, 1, 4)

        q = q[0]
        k, v = kv[0], kv[1]

        if self._use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                x = F.scaled_dot_product_attention(
                    q, k, v, dropout_p=self._proj_drop_prob)
                attn = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self._scale
            attn = attn.softmax(dim=-1)
            x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, I, C)
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


class MixedBlock(nn.Module):
    def __init__(
        self,
        *args,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        qk_scale: Union[float, None] = None,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs
    ) -> None:
        super(MixedBlock, self).__init__(*args, **kwargs)

        self._norm1 = norm_layer(dim)
        self._attn = MixedAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self._norm2 = norm_layer(dim)
        hidden_mlp_dim = int(dim * mlp_ratio)
        self._mlp = MLP(
            in_features=dim,
            hidden_features=hidden_mlp_dim,
            act_layer=act_layer,
            drop=drop
        )

    def forward(
        self,
        image_features: torch.Tensor,
        language_features: torch.Tensor,
        return_attention: bool = False, mask:
        Union[None, torch.Tensor] = None
    ) -> torch.Tensor:
        y, attn = self._attn(
            image_features=self._norm1(image_features),
            language_features=self._norm2(language_features),
            mask=mask
        )
        if return_attention:
            return attn
        image_features = image_features + y
        image_features = image_features + self._mlp(self._norm2(image_features))
        return image_features


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=12,
        qkv_bias=False,
        use_sdpa=True
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, int(dim*2), bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_sdpa = use_sdpa

    def forward(self, q, x):
        B, n, C = q.shape
        q = self.q(q).reshape(B, n, self.num_heads, C //
                              self.num_heads).permute(0, 2, 1, 3)

        B, N, C = x.shape
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C //
                                self.num_heads).permute(2, 0, 3, 1, 4)
        # (batch_size, num_heads, seq_len, feature_dim_per_head)
        k, v = kv[0], kv[1]

        if self.use_sdpa:
            with torch.backends.cuda.sdp_kernel():
                q = F.scaled_dot_product_attention(q, k, v)
        else:
            xattn = (q @ k.transpose(-2, -1)) * self.scale
            # (batch_size, num_heads, query_len, seq_len)
            xattn = xattn.softmax(dim=-1)
            q = (xattn @ v)

        q = q.transpose(1, 2).reshape(B, n, C)
        q = self.proj(q)

        return q


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.xattn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, q, x):
        y = self.xattn(q, self.norm1(x))
        q = q + y
        q = q + self.mlp(self.norm2(q))
        return q
