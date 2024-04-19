from __future__ import annotations

import loralib
import torch
import torch.nn as nn


from typing import List, Tuple, Union


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
