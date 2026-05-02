import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class MLP(nn.Module):
    """Standard dense multi-layer perceptron"""

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(d_hidden, d_model))
        torch.nn.init.normal_(self.W_in, std=0.02)

        self.b_in = nn.Parameter(torch.zeros(d_hidden))
        self.relu = nn.ReLU()

        self.W_out = nn.Parameter(torch.empty(d_model, d_hidden))
        torch.nn.init.normal_(self.W_out, std=0.02)

    def forward(
        self, x: Float[Tensor, "batch n_seq d_model"]
    ) -> Float[Tensor, "batch n_seq d_model"]:
        return self.relu(x @ self.W_in.T + self.b_in) @ self.W_out.T
