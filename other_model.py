import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class BiLSTMModel(nn.Module):
    def __init__(self, input_len=200, d_model=256, out_num=4, num_layers=3):
        super().__init__()

        self.proj = nn.Linear(input_len, d_model)

        # 双向LSTM层
        self.lstm1 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                             bidirectional=True, batch_first=True, dropout=0.1)
        self.lstm2 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                             bidirectional=True, batch_first=True, dropout=0.1)

        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)

        # 全连接输出层
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model * 4)
        self.fc3 = nn.Linear(d_model * 4, d_model)
        self.fc4 = nn.Linear(d_model, d_model // 2)
        self.out_layer = nn.Linear(d_model // 2, out_num)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 6, input_len]

        # 投影到特征空间
        x = self.proj(x)  # [B, 6, d_model]

        # LSTM编码
        lstm1_out, _ = self.lstm1(x)  # [B, 6, d_model]
        lstm1_out = self.norm(lstm1_out)

        lstm2_out, _ = self.lstm2(lstm1_out)  # [B, 6, d_model]
        x = x + lstm2_out  # 残差连接
        x = self.norm(x)

        # 聚合维度
        x = x.mean(dim=1)  # [B, d_model]

        # 深层全连接网络with残差连接
        x_input = x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = x + x_input  # 残差连接

        x = self.relu(self.fc4(x))
        x = self.out_layer(x)  # [B, out_num]

        return x


class PositionalEncoding(nn.Module):
    """标准的正弦位置编码"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        pos_emb = self.pe[:seq_len, :, :].transpose(0, 1)  # [1, seq_len, d_model]
        return x + pos_emb.expand_as(x)


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, input_len=200, d_model=256, nhead=8, num_layers=6, out_num=4):
        super().__init__()

        # 投影层
        self.proj = nn.Linear(input_len, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len=6)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 深度全连接网络
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 4, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 2, d_model),
            nn.GELU()
        )

        # 残差投影
        self.residual_proj = nn.Linear(d_model, d_model)

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, out_num)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [B, 6, input_len]

        # 投影到Transformer维度
        x = self.proj(x)  # [B, 6, d_model]

        # 添加位置编码
        x = self.pos_encoding(x)

        # Transformer Encoder处理
        encoded = self.transformer_encoder(x)  # [B, 6, d_model]

        # 聚合维度
        aggregated = encoded.mean(dim=1)  # [B, d_model]

        # 深度全连接处理
        x_residual = self.residual_proj(aggregated)
        x = self.fc_layers(aggregated)
        x = x + x_residual  # 残差连接

        # 输出预测
        output = self.output_layers(x)  # [B, out_num]

        return output


class UNet(nn.Module):
    """Standard 1D U-Net: 仅使用输入 x 的第 3 列 (index=2)，输入视为 [B,6,L]，网络实际用 [B,1,L]，输出 [B,4]"""
    def __init__(self, input_len, out_num=4):
        super().__init__()
        self.length = input_len
        x = 32
        in_channels = 1  # 固定为 1

        self.down1 = self._double_conv(in_channels, x)
        self.pool1 = nn.MaxPool1d(2)

        self.down2 = self._double_conv(x, x * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.down3 = self._double_conv(x * 2, x * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.down4 = self._double_conv(x * 4, x * 8)
        self.pool4 = nn.MaxPool1d(2)

        self.bottom = self._double_conv(x * 8, x * 16)

        self.up6 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.up_conv6 = self._double_conv(x * 16 + x * 8, x * 8)

        self.up7 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.up_conv7 = self._double_conv(x * 8 + x * 4, x * 4)

        self.up8 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.up_conv8 = self._double_conv(x * 4 + x * 2, x * 2)

        self.up9 = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.up_conv9 = self._double_conv(x * 2 + x, x)

        self.out_conv = nn.Conv1d(x, 1, kernel_size=1)
        self.head = nn.Linear(1, out_num)

    @staticmethod
    def _align_1d(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        tx = x.size(-1)
        tr = ref.size(-1)
        if tx == tr:
            return x
        if tx > tr:
            return x[..., :tr]
        return F.pad(x, (0, tr - tx))

    def _double_conv(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 支持两种输入:
        # 1) [B,6,L] -> 取第 3 列
        # 2) [B,1,L] -> 直接使用
        if x.dim() != 3:
            raise ValueError(f"UNet 期望输入维度为 3 (B,C,L)，但得到: {tuple(x.shape)}")

        if x.size(1) == 6:
            x = x[:, 2:3, :]  # 第 3 列，保持通道维: [B,1,L]
        elif x.size(1) == 1:
            pass
        else:
            raise ValueError(f"UNet 期望通道数为 6 或 1，但得到: {int(x.size(1))}")

        conv1 = self.down1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.down4(pool3)
        pool4 = self.pool4(conv4)

        conv5 = self.bottom(pool4)

        up6 = self._align_1d(self.up6(conv5), conv4)
        merge6 = torch.cat([up6, conv4], dim=1)
        conv6 = self.up_conv6(merge6)

        up7 = self._align_1d(self.up7(conv6), conv3)
        merge7 = torch.cat([up7, conv3], dim=1)
        conv7 = self.up_conv7(merge7)

        up8 = self._align_1d(self.up8(conv7), conv2)
        merge8 = torch.cat([up8, conv2], dim=1)
        conv8 = self.up_conv8(merge8)

        up9 = self._align_1d(self.up9(conv8), conv1)
        merge9 = torch.cat([up9, conv1], dim=1)
        conv9 = self.up_conv9(merge9)

        out = self.out_conv(conv9)   # [B,1,L]
        gap = out.mean(dim=2)        # [B,1]
        return self.head(gap)        # [B,4]
