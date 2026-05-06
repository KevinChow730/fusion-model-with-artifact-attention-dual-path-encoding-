import torch
import torch.nn as nn
from encode_decode import EncodeBlock, DecodeBlock
from diff_attention import CrossDiffAttention


class TaskHead(nn.Module):
    """
    输入: 解码特征 [B, 2, input_dim]
    输出: [B, out_num]，默认 out_num=1
    结构与原 FFN 保持一致，但作为任务专属头使用
    """
    def __init__(self, input_dim: int, d_model: int, out_num: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(2 * input_dim, d_model)
        self.fc_mid1 = nn.Linear(d_model, d_model * 3)
        self.fc_mid2 = nn.Linear(d_model * 3, d_model * 6)
        self.fc_mid3 = nn.Linear(d_model * 6, d_model * 2)
        self.fc_mid4 = nn.Linear(d_model * 2, d_model)
        self.out_layer = nn.Linear(d_model, out_num)

    def forward(self, dec: torch.Tensor) -> torch.Tensor:
        # dec: [B, 2, input_dim]
        dec_flat = dec.reshape(dec.size(0), -1)               # [B, 2*input_dim]
        x_fc1 = self.relu(self.fc1(dec_flat))                 # [B, d_model]
        x_fc_mid1 = self.relu(self.fc_mid1(x_fc1))
        x_fc_mid2 = self.relu(self.fc_mid2(x_fc_mid1))
        x_fc_mid3 = self.relu(self.fc_mid3(x_fc_mid2))
        out = self.relu(self.fc_mid4(x_fc_mid3))              # [B, d_model]
        x = x_fc1 + out
        return self.out_layer(self.relu(x))                   # [B, out_num]


class Trans_wo(nn.Module):
    def __init__(self, input_dim=64, d_model=256, out_num=1, num_heads=16):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.enc_a = EncodeBlock(input_len=input_dim, d_model=d_model)
        self.enc_b = EncodeBlock(input_len=input_dim, d_model=d_model)  # 保留以兼容输入形状
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn = CrossDiffAttention(
            dim=d_model, num_heads=num_heads, qkv_bias=True,
            attn_drop=0.0, proj_drop=0.0, lambda_init=0.8
        )
        self.dec_SBP = DecodeBlock(input_len=input_dim, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_len=input_dim, d_model=d_model)
        self.dec_HR = DecodeBlock(input_len=input_dim, d_model=d_model)
        self.head_SBP = TaskHead(input_dim=input_dim, d_model=d_model, out_num=out_num)
        self.head_DBP = TaskHead(input_dim=input_dim, d_model=d_model, out_num=out_num)
        self.head_HR = TaskHead(input_dim=input_dim, d_model=d_model, out_num=out_num)

    def forward(self, x):
        assert x.dim() == 3 and x.size(1) == 4 and x.size(2) == self.input_dim, "输入形状应为 [B, 4, input_dim]"
        xa = x[:, 0:2, :]          # 使用 A 分支
        xb = x[:, 2:4, :]          # 仍接收但不用于注意力（消融 B 影响）

        a_encoded = self.enc_a(xa)
        b_encoded = self.enc_b(xb)  # 计算但不参与后续注意力
        a_encoded = self.norm1(a_encoded)
        b_encoded = self.norm2(b_encoded)

        a_residual = a_encoded
        attn_output = self.cross_attn(a_encoded, a_encoded)  # 消融: 仅用 A 做自注意力
        shared_latent = a_residual + attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_HR], dim=1)
