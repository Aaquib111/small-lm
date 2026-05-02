import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class MultiHeadAttention(nn.Module):
    """Standard multi-head attention with RoPE"""

    def __init__(self, n_head: int, max_ctx: int, d_model: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv_proj = nn.Parameter(torch.empty(3 * d_model, d_model))
        torch.nn.init.normal_(self.qkv_proj, std=0.02)
        self.qkv_bias = nn.Parameter(torch.zeros(3 * d_model))

        self.W_out = nn.Parameter(torch.empty(d_model, d_model))
        torch.nn.init.normal_(self.W_out, std=0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))

        theta: Float[Tensor, "seq d_head/2"] = (
            torch.arange(max_ctx)[:, None]
            @ (1e5 ** (-2 * torch.arange(self.d_head // 2) / self.d_head))[None, :]
        )
        sin_rot = torch.sin(theta)
        cos_rot = torch.cos(theta)
        self.register_buffer("sin_rot", sin_rot)
        self.register_buffer("cos_rot", cos_rot)

    def forward(
        self, x: Float[Tensor, "batch n_seq d_model"]
    ) -> Float[Tensor, "batch n_seq d_model"]:
        batch, n_seq, d_model = x.shape
        d_head = d_model // self.n_head
        qkv: Float[Tensor, "batch n_seq 3_d_model"] = (
            (x @ self.qkv_proj) + self.qkv_bias
        ).reshape(batch, n_seq, 3, self.n_head, d_head)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (batch n_head seq d_head)

        q = q.reshape(batch, self.n_head, n_seq, d_head // 2, 2)
        k = k.reshape(batch, self.n_head, n_seq, d_head // 2, 2)

        q_rot: Float[Tensor, "batch n_head n_seq d_head"] = (
            q[..., 0] * self.cos_rot - q[..., 1] * self.sin_rot
        ).reshape(batch, self.n_head, n_seq, d_head)
        k_rot: Float[Tensor, "batch n_head n_seq d_head"] = (
            k[..., 0] * self.cos_rot - k[..., 1] * self.sin_rot
        ).reshape(batch, self.n_head, n_seq, d_head)

        # MHA
        attn_pttn = q_rot @ k_rot.T
        attn_pttn.masked_fill_(
            torch.triu(torch.ones_like(attn_pttn), diagonal=1).bool(), -torch.inf
        )
        z: Float[Tensor, "batch n_head q_seq k_seq"] = (
            nn.functional.softmax(attn_pttn) / torch.sqrt(torch.tensor(self.d_head)) @ v
        )

        return (z @ self.W_out.T) + self.b_out
