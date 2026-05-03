"""Pre-ln transformer block implementation."""

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from src.components.attention.mha import MultiHeadAttention
from src.components.mlp.dense import MLP
from src.components.norms import RMSNorm


class TransformerBlock(nn.Module):
    """Pre-ln transformer block"""

    def __init__(self, n_head: int, max_ctx: int, d_model: int, d_hidden: int):
        super().__init__()
        self.attn = MultiHeadAttention(n_head, max_ctx, d_model)
        self.mlp = MLP(d_model, d_hidden)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

    def forward(
        self, x: Float[Tensor, "batch n_seq d_model"]
    ) -> Float[Tensor, "batch n_seq d_model"]:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
