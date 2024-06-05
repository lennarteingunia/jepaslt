from __future__ import annotations
import math

import torch


class Linear(torch.nn.Linear):

    def __init__(
        self: Linear,
        in_features: int,
        out_features: int,
        bias: bool = True,
        rank: int = 0,
        lora_alpha: int = 1,
        merge_weights: bool = True,
        **kwargs
    ) -> None:
        
        super(Linear, self).__init__(
            in_features=in_features, 
            out_features=out_features,
            bias=bias,
            **kwargs
        )

        self.rank = rank
        self.merge_weights = merge_weights
        self.merged = False

        if rank > 0:
            self.lora_A = torch.nn.Parameter(self.weight.new_zeros((rank, in_features)))
            self.lora_B = torch.nn.Parameter(self.weight.new_zeros((out_features, rank)))
            self.scaling = lora_alpha / rank
            
            # Freezing inner weights
            self.weight.requires_grad = False
            
            if bias:
                self.bias.requires_grad = False

    @property
    def lora(self: Linear) -> torch.Tensor:
        return self.lora_B @ self.lora_A * self.scaling

    def reset_parameters(self: Linear) -> None:
        super(Linear, self).reset_parameters()
        if self.rank > 0:
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True) -> None:
        super(Linear, self).train(mode=mode)

        if mode and self.merge_weights:
            self.merge(mode=False)
        elif not mode and self.merge_weights:
            self.merge(mode=True)

    def forward(self: Linear, x: torch.Tensor) -> torch.Tensor:
        result = torch.nn.functional.linear(x, self.weight, bias=self.bias)
        if self.rank > 0 and not self.merged:
            result += torch.nn.functional.linear(x, self.lora, bias=None)
        return result
    
    def forward_linear(self: Linear, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return torch.nn.functional.linear(x, self.weight - self.lora, bias=self.bias)
        else:
            return torch.nn.functional.linear(x, self.weight, bias=self.bias)

    def forward_lora(self: Linear, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.lora, bias=None)
    
    def merge(self: Linear, mode: bool = True) -> None:
        if self.rank > 0:
            if mode and not self.merged:
                self.weight.data += self.lora
                self.merged = True
            elif not mode and self.merged:
                self.weight.data -= self.lora
                self.merged = False