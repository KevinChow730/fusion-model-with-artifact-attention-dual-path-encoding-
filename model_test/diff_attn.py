# `diff_attention.py`
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output


class CrossDiffAttention(nn.Module):
    """修改的差分注意力，支持交叉注意力（Q来自第一列，K来自第二列）"""

    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0., lambda_init=0.8):
        super().__init__()
        if num_heads % 2 != 0:
            raise ValueError("num_heads must be even for Differential Attention.")
        self.dim = dim
        self.num_heads = num_heads
        self.effective_heads = num_heads // 2
        self.head_dim = dim // num_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.diff_norm = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_init = lambda_init

    def forward(self, q_input: torch.Tensor, k_input: torch.Tensor, return_attn: bool = False):
        """
        Args:
            q_input: [B, N, dim]
            k_input: [B, N, dim]
            return_attn: 是否返回差分前/后注意力张量（用于可视化）
        Returns:
            if return_attn == False:
                x_out: [B, N, dim]
            else:
                x_out: [B, N, dim]
                attn_dict: {
                    `attn1`: [B, effective_heads, N, N],
                    `attn2`: [B, effective_heads, N, N],
                    `diff_attn`: [B, effective_heads, N, N],
                    `lambda_full`: 标量张量
                }
        """
        B, N, _ = q_input.shape

        q = self.q_proj(q_input)  # [B, N, dim]
        k = self.k_proj(k_input)  # [B, N, dim]
        v = self.v_proj(k_input)  # [B, N, dim]

        q = q.view(B, N, 2 * self.effective_heads, self.head_dim).transpose(1, 2)  # [B, 2Eh, N, Dh]
        k = k.view(B, N, 2 * self.effective_heads, self.head_dim).transpose(1, 2)  # [B, 2Eh, N, Dh]
        v = v.view(B, N, self.effective_heads, 2 * self.head_dim).transpose(1, 2)  # [B, Eh, N, 2Dh]

        q = q * self.scaling

        attn_scores = torch.matmul(q, k.transpose(-1, -2))  # [B, 2Eh, N, N]
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_drop(attn_probs)

        attn_probs = attn_probs.view(B, self.effective_heads, 2, N, N)  # [B, Eh, 2, N, N]

        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        attn1 = attn_probs[:, :, 0, :, :]  # [B, Eh, N, N]
        attn2 = attn_probs[:, :, 1, :, :]  # [B, Eh, N, N]
        diff_attn = attn1 - lambda_full * attn2  # [B, Eh, N, N]

        attn_output = torch.matmul(diff_attn, v)  # [B, Eh, N, 2Dh]
        attn_output = self.diff_norm(attn_output) * (1 - self.lambda_init)

        attn_output = attn_output.transpose(1, 2).reshape(B, N, 2 * self.effective_heads * self.head_dim)  # [B, N, dim]
        x_out = self.out_proj(attn_output)
        x_out = self.proj_drop(x_out)

        if not return_attn:
            return x_out

        attn_dict = {
            "attn1": attn1.detach(),
            "attn2": attn2.detach(),
            "diff_attn": diff_attn.detach(),
            "lambda_full": lambda_full.detach(),
        }
        return x_out, attn_dict
