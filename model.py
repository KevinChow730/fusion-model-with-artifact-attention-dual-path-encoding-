import torch
import torch.nn as nn
from encode_decode import EncodeBlock, DecodeBlock
from diff_attention import CrossDiffAttention


class TaskHead(nn.Module):
    def __init__(self, input_dim: int, d_model: int, out_num: int = 1, drop_p: float = 0):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)

        self.fc1 = nn.Linear(2 * input_dim, d_model)
        self.fc_mid1 = nn.Linear(d_model, d_model * 3)
        self.fc_mid2 = nn.Linear(d_model * 3, d_model * 6)
        self.fc_mid3 = nn.Linear(d_model * 6, d_model * 2)
        self.fc_mid4 = nn.Linear(d_model * 2, d_model)
        self.out_layer = nn.Linear(d_model, out_num)

    def forward(self, dec: torch.Tensor) -> torch.Tensor:
        dec_flat = dec.reshape(dec.size(0), -1)          # [B, 2*input_dim]
        x = self.relu(self.fc1(dec_flat))                # [B, d_model]
        x_fc1 = x

        x = self.dropout(self.relu(self.fc_mid1(x)))
        x = self.dropout(self.relu(self.fc_mid2(x)))
        x = self.dropout(self.relu(self.fc_mid3(x)))
        out = self.relu(self.fc_mid4(x))                 # [B, d_model]

        x = x_fc1 + out                                  # 残差
        x = self.relu(x)
        x = self.dropout(x)
        return self.out_layer(x)                         # [B, out_num]


class Trans(nn.Module):
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_ir = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlock(input_dim=input_len, d_model=d_model)

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.cross_attn = CrossDiffAttention(
            dim=d_model, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop_p, proj_drop=drop_p, lambda_init=0.8
        )

        # 编码后和 cross\-attn 后的 dropout
        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        self.dec_SBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_dim=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]
        pressure = x[:, 4:6, :]

        enc_red = self.enc_red(red)            # [B, 1, d_model]
        enc_ir = self.enc_ir(ir)
        pressure_encoded = self.enc_pressure(pressure)

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)

        ppg_encoded = alpha * enc_red + beta * enc_ir
        ppg_encoded = self.norm1(ppg_encoded)
        pressure_encoded = self.norm2(pressure_encoded)

        # 编码后做一次 dropout
        ppg_encoded = self.dropout_enc(ppg_encoded)
        pressure_encoded = self.dropout_enc(pressure_encoded)

        ppg_res = ppg_encoded
        attn_output = self.cross_attn(ppg_encoded, pressure_encoded)  # [B, 1, d_model]
        attn_output = self.dropout_attn(attn_output)

        shared_latent = ppg_res + attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_HR(dec_feat_SpO2)
        y_HR = self.head_SpO2(dec_feat_HR)

        y = torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)
        return y

    class Simple(nn.Module):
        def __init__(self, input_len: int = 200, num_channels: int = 6,
                     hidden_dim1: int = 512, hidden_dim2: int = 256, out_dim: int = 4, drop_p: float = 0.0):
            super().__init__()
            self.input_len = input_len
            self.num_channels = num_channels
            in_dim = num_channels * input_len  # 6 * 200 = 1200

            self.fc1 = nn.Linear(in_dim, hidden_dim1)
            self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
            self.fc3 = nn.Linear(hidden_dim2, out_dim)

            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=drop_p)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: [B, 6, 200]，保持和原 Trans 模型一致
            assert x.dim() == 3 and x.size(1) == self.num_channels and x.size(2) == self.input_len, \
                "输入形状应为 [B, 6, input_len]"

            # 展平为 [B, 1200]
            x = x.reshape(x.size(0), -1)

            x = self.relu(self.fc1(x))
            x = self.dropout(x)

            x = self.relu(self.fc2(x))
            x = self.dropout(x)

            # 输出 [B, 4]，对应 SBP, DBP, SpO2, HR 四个任务
            out = self.fc3(x)
            return out


class TransNoPressDifAttn(nn.Module):
    """
    方案A: 输入仍为 [B, 6, L]，但不使用 pressure 融合（不走 cross-attn）。
    输出仍为 [B, 4]（按 out_num 拼接）。
    """
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_ir = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlock(input_dim=input_len, d_model=d_model)  # 保留以保证结构/参数接口一致

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)  # 保留

        self.cross_attn = CrossDiffAttention(
            dim=d_model, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop_p, proj_drop=drop_p, lambda_init=0.8
        )  # 保留但 forward 不用

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        self.dec_SBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_dim=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]

        enc_red = self.enc_red(red)  # [B, 1, d_model]
        enc_ir = self.enc_ir(ir)

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)

        ppg_encoded = alpha * enc_red + beta * enc_ir
        ppg_encoded = self.norm1(ppg_encoded)
        ppg_encoded = self.dropout_enc(ppg_encoded)

        shared_latent = ppg_encoded

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_SpO2(dec_feat_SpO2)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)


class TransNoPressure(nn.Module):
    """
    方案B: 输入仍为 [B, 6, L]，保留 cross-attn，但用 ppg_encoded 作为 key/value，
    即 pressure 信息不进入融合，结构/正则形式保留。
    输出仍为 [B, 4]（按 out_num 拼接）。
    """
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_ir = EncodeBlock(input_dim=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlock(input_dim=input_len, d_model=d_model)  # 保留但 forward 不用其输出

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)  # 保留

        self.cross_attn = CrossDiffAttention(
            dim=d_model, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop_p, proj_drop=drop_p, lambda_init=0.8
        )

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        self.dec_SBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_dim=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_dim=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_dim=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]

        enc_red = self.enc_red(red)  # [B, 1, d_model]
        enc_ir = self.enc_ir(ir)

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)

        ppg_encoded = alpha * enc_red + beta * enc_ir
        ppg_encoded = self.norm1(ppg_encoded)
        ppg_encoded = self.dropout_enc(ppg_encoded)

        ppg_res = ppg_encoded
        kv = ppg_encoded  # 用 ppg 作为 key/value
        attn_output = self.cross_attn(ppg_encoded, kv)  # [B, 1, d_model]
        attn_output = self.dropout_attn(attn_output)

        shared_latent = ppg_res + attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_SpO2(dec_feat_SpO2)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)
