"""Embedding layer"""

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(vocab_size, d_model))
        torch.nn.init.normal_(self.embedding, std=0.02)

    def forward(
        self, x: Int[Tensor, "batch n_seq"]
    ) -> Float[Tensor, "batch n_seq d_model"]:
        return self.embedding[x]
