"""RMSNorm implementation"""

import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()

        self.scale = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(
        self, x: Float[Tensor, "batch n_seq d_model"]
    ) -> Float[Tensor, "batch n_seq d_model"]:
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
        x_normed = x / rms
        return x_normed * self.scale
