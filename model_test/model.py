import torch
import torch.nn as nn
import torch.nn.functional as F
from encode_decode import EncodeBlock, DecodeBlock, EncodeBlockSTOnly
from diff_attention import CrossDiffAttention


class TaskHead(nn.Module):
    """
    适配 DecodeBlock 输出:
      输入:  dec [B, 2, L]
      输出:  y   [B, out_num]

    做法:
      1) 对时间维 L 做全局平均池化: [B, 2, L] -> [B, 2]
      2) MLP 回归到 out_num
    """
    def __init__(self, input_len: int, d_model: int, out_num: int = 1, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len  # 输入序列长度
        self.d_model = d_model
        self.out_num = out_num

        self.pool = nn.AdaptiveAvgPool1d(1)  # [B, 2, L] -> [B, 2, 1]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_p)

        self.fc1 = nn.Linear(2, d_model)
        self.fc_mid1 = nn.Linear(d_model, d_model * 3)
        self.fc_mid2 = nn.Linear(d_model * 3, d_model * 6)
        self.fc_mid3 = nn.Linear(d_model * 6, d_model * 2)
        self.fc_mid4 = nn.Linear(d_model * 2, d_model)
        self.out_layer = nn.Linear(d_model, out_num)

    def forward(self, dec: torch.Tensor) -> torch.Tensor:
        assert dec.dim() == 3 and dec.size(1) == 2, r"TaskHead 输入应为 \[B, 2, L\]"

        x = self.pool(dec).squeeze(-1)  # [B, 2]
        x = self.relu(self.fc1(x))      # [B, d_model]
        x_fc1 = x

        x = self.dropout(self.relu(self.fc_mid1(x)))
        x = self.dropout(self.relu(self.fc_mid2(x)))
        x = self.dropout(self.relu(self.fc_mid3(x)))
        out = self.relu(self.fc_mid4(x))  # [B, d_model]

        x = x_fc1 + out                   # 残差
        x = self.relu(x)
        x = self.dropout(x)
        return self.out_layer(x)          # [B, out_num]


class FADE(nn.Module):
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlock(input_len=input_len, d_model=d_model)
        self.enc_ir = EncodeBlock(input_len=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlock(input_len=input_len, d_model=d_model)

        # 以 encoder 的输出维作为后续主干维度
        self.c_feat = self.enc_red.c_feat

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # LayerNorm 改为归一化 c_feat（匹配 [B, L, c_feat]）
        self.norm1 = nn.LayerNorm(self.c_feat)
        self.norm2 = nn.LayerNorm(self.c_feat)

        # CrossDiffAttention 的 dim 也用 c_feat（不再用 d_model）
        self.cross_attn = CrossDiffAttention(
            dim=self.c_feat, num_heads=num_heads, qkv_bias=True,
            attn_drop=drop_p, proj_drop=drop_p, lambda_init=0.8
        )

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        # 可学习的加权相减系数（标量，forward 里会 reshape 做广播）
        self.w_ppg = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_attn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # DecodeBlock 已经期望输入 [B, L, c_feat]，无需改
        self.dec_SBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_len=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]
        pressure = x[:, 4:6, :]

        # EncodeBlock 输出: [B, L, c_feat]
        enc_red = self.enc_red(red)
        enc_ir = self.enc_ir(ir)
        pressure_encoded = self.enc_pressure(pressure)

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)
        ppg_encoded = alpha * enc_red + beta * enc_ir  # [B, L, c_feat]

        # 直接在 c_feat 维度上做 norm / attn（不投影到 d_model）
        ppg_encoded = self.norm1(ppg_encoded)
        pressure_encoded = self.norm2(pressure_encoded)

        ppg_encoded = self.dropout_enc(ppg_encoded)
        pressure_encoded = self.dropout_enc(pressure_encoded)

        ppg_res = ppg_encoded
        attn_output = self.cross_attn(ppg_encoded, pressure_encoded)  # [B, L, c_feat]
        attn_output = self.dropout_attn(attn_output)

        w_ppg = self.w_ppg.view(1, 1, 1)
        w_attn = self.w_attn.view(1, 1, 1)
        shared_latent = w_ppg * ppg_res - w_attn * attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_HR(dec_feat_SpO2)
        y_HR = self.head_SpO2(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)


class FDE(nn.Module):
    """
    输入仍为 [B, 6, L]，仍然走 pressure 编码通道，但不使用 diff-attn 融合；
    使用普通交叉注意力融合。
    输出仍为 [B, 4]（按 out_num 拼接）。
    """
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlock(input_len=input_len, d_model=d_model)
        self.enc_ir = EncodeBlock(input_len=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlock(input_len=input_len, d_model=d_model)

        # 以 encoder 的输出维作为后续主干维度
        self.c_feat = self.enc_red.c_feat

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # LayerNorm 改为归一化 c_feat（匹配 [B, L, c_feat]）
        self.norm1 = nn.LayerNorm(self.c_feat)
        self.norm2 = nn.LayerNorm(self.c_feat)

        # 使用标准的多头注意力替代diff-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.c_feat,
            num_heads=num_heads,
            dropout=drop_p,
            batch_first=True
        )

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        # 可学习的加权相减系数（标量，forward 里会 reshape 做广播）
        self.w_ppg = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_attn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        # DecodeBlock 已经期望输入 [B, L, c_feat]，无需改
        self.dec_SBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_len=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]
        pressure = x[:, 4:6, :]

        # EncodeBlock 输出: [B, L, c_feat]
        enc_red = self.enc_red(red)
        enc_ir = self.enc_ir(ir)
        pressure_encoded = self.enc_pressure(pressure)

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)
        ppg_encoded = alpha * enc_red + beta * enc_ir  # [B, L, c_feat]

        # 直接在 c_feat 维度上做 norm / attn（不投影到 d_model）
        ppg_encoded = self.norm1(ppg_encoded)
        pressure_encoded = self.norm2(pressure_encoded)

        ppg_encoded = self.dropout_enc(ppg_encoded)
        pressure_encoded = self.dropout_enc(pressure_encoded)

        ppg_res = ppg_encoded
        # 使用标准交叉注意力: query=ppg_encoded, key=value=pressure_encoded
        attn_output, _ = self.cross_attn(ppg_encoded, pressure_encoded, pressure_encoded)  # [B, L, c_feat]
        attn_output = self.dropout_attn(attn_output)

        w_ppg = self.w_ppg.view(1, 1, 1)
        w_attn = self.w_attn.view(1, 1, 1)
        shared_latent = w_ppg * ppg_res - w_attn * attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_SpO2(dec_feat_SpO2)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)


class FE(nn.Module):
    """
    编码器为短时路径（EncodeBlockSTOnly），用普通 cross-attn 融合：
    Q=ppg_encoded, K,V=pressure_encoded。
    输入:  [B, 6, L]
    输出:  [B, 4]（按 out_num 拼接）
    """
    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlockSTOnly(input_len=input_len, d_model=d_model)
        self.enc_ir = EncodeBlockSTOnly(input_len=input_len, d_model=d_model)
        self.enc_pressure = EncodeBlockSTOnly(input_len=input_len, d_model=d_model)

        # 获取c_feat维度
        self.c_feat = self.enc_red.c_feat

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        self.norm_ppg = nn.LayerNorm(self.c_feat)
        self.norm_press = nn.LayerNorm(self.c_feat)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.c_feat,
            num_heads=num_heads,
            dropout=drop_p,
            batch_first=True
        )

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        # 可学习的加权相减系数（标量，forward 里会 reshape 做广播）
        self.w_ppg = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_attn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.dec_SBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_len=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]
        pressure = x[:, 4:6, :]

        enc_red = self.enc_red(red)                      # [B, L, c_feat]
        enc_ir = self.enc_ir(ir)                         # [B, L, c_feat]
        pressure_encoded = self.enc_pressure(pressure)   # [B, L, c_feat]

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)
        ppg_encoded = alpha * enc_red + beta * enc_ir    # [B, L, c_feat]

        ppg_encoded = self.norm_ppg(ppg_encoded)
        pressure_encoded = self.norm_press(pressure_encoded)

        ppg_encoded = self.dropout_enc(ppg_encoded)
        pressure_encoded = self.dropout_enc(pressure_encoded)
        ppg_res = ppg_encoded

        # 普通交叉注意力：Q=ppg, K/V=press
        attn_output, _ = self.cross_attn(ppg_encoded, pressure_encoded, pressure_encoded)  # [B, L, c_feat]
        attn_output = self.dropout_attn(attn_output)

        w_ppg = self.w_ppg.view(1, 1, 1)
        w_attn = self.w_attn.view(1, 1, 1)
        shared_latent = w_ppg * ppg_res - w_attn * attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_SpO2(dec_feat_SpO2)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)


class FEwoP(nn.Module):
    """
    不使用pressure编码通道，仅用红外和红光编码的ppg特征做自注意力融合：
    输入:  [B, 6, L]
    输出:  [B, 4]（按 out_num 拼接）
    """

    def __init__(self, input_len=200, d_model=256, out_num=1, num_heads=16, drop_p: float = 0):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        self.enc_red = EncodeBlockSTOnly(input_len=input_len, d_model=d_model)
        self.enc_ir = EncodeBlockSTOnly(input_len=input_len, d_model=d_model)
        # 移除enc_pressure，不再使用pressure信息

        # 获取c_feat维度
        self.c_feat = self.enc_red.c_feat

        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # 只需要ppg的LayerNorm
        self.norm_ppg = nn.LayerNorm(self.c_feat)

        # 使用自注意力而不是交叉注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.c_feat,
            num_heads=num_heads,
            dropout=drop_p,
            batch_first=True
        )

        self.dropout_enc = nn.Dropout(p=drop_p)
        self.dropout_attn = nn.Dropout(p=drop_p)

        # 可学习的加权相减系数（标量，forward 里会 reshape 做广播）
        self.w_ppg = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.w_attn = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.dec_SBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_DBP = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_SpO2 = DecodeBlock(input_len=input_len, d_model=d_model)
        self.dec_HR = DecodeBlock(input_len=input_len, d_model=d_model)

        self.head_SBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_DBP = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_SpO2 = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)
        self.head_HR = TaskHead(input_len=input_len, d_model=d_model, out_num=out_num, drop_p=drop_p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 6 and x.size(2) == self.input_len, \
            "输入形状应为 [B, 6, input_len]"

        red = x[:, 0:2, :]
        ir = x[:, 2:4, :]
        # 移除pressure处理，不再使用pressure信息

        enc_red = self.enc_red(red)    # [B, L, c_feat]
        enc_ir = self.enc_ir(ir)       # [B, L, c_feat]

        alpha = self.alpha.view(1, 1, 1)
        beta = self.beta.view(1, 1, 1)
        ppg_encoded = alpha * enc_red + beta * enc_ir    # [B, L, c_feat]

        ppg_encoded = self.norm_ppg(ppg_encoded)
        ppg_encoded = self.dropout_enc(ppg_encoded)
        ppg_res = ppg_encoded

        # 自注意力：Q=K=V=ppg_encoded
        attn_output, _ = self.self_attn(ppg_encoded, ppg_encoded, ppg_encoded)  # [B, L, c_feat]
        attn_output = self.dropout_attn(attn_output)

        w_ppg = self.w_ppg.view(1, 1, 1)
        w_attn = self.w_attn.view(1, 1, 1)
        shared_latent = w_ppg * ppg_res - w_attn * attn_output

        dec_feat_SBP = self.dec_SBP(shared_latent)
        dec_feat_DBP = self.dec_DBP(shared_latent)
        dec_feat_SpO2 = self.dec_SpO2(shared_latent)
        dec_feat_HR = self.dec_HR(shared_latent)

        y_SBP = self.head_SBP(dec_feat_SBP)
        y_DBP = self.head_DBP(dec_feat_DBP)
        y_SpO2 = self.head_SpO2(dec_feat_SpO2)
        y_HR = self.head_HR(dec_feat_HR)

        return torch.cat([y_SBP, y_DBP, y_SpO2, y_HR], dim=1)