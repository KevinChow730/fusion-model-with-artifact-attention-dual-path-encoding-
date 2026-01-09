import torch
import torch.nn as nn
import math


class CNNModel(nn.Module):
    def __init__(self, input_dim=128, d_model=256, out_num=2):
        super().__init__()

        # 分别为两列构建CNN编码器
        self.col1_proj = nn.Linear(input_dim, d_model)
        self.col2_proj = nn.Linear(input_dim, d_model)

        # 多尺度CNN特征提取
        # 第一列CNN路径
        self.col1_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.col1_conv2 = nn.Conv1d(d_model, d_model * 2, kernel_size=5, padding=2)
        self.col1_conv3 = nn.Conv1d(d_model, d_model * 2, kernel_size=7, padding=3)

        # 第二列CNN路径
        self.col2_conv1 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.col2_conv2 = nn.Conv1d(d_model, d_model * 2, kernel_size=5, padding=2)
        self.col2_conv3 = nn.Conv1d(d_model, d_model * 2, kernel_size=7, padding=3)

        # 修正特征融合层维度
        # 每列特征：d_model + d_model*2 + d_model*2 = d_model*5
        # 两列总共：d_model*5 * 2 = d_model*10
        fusion_dim = d_model * 10
        self.fusion_conv1 = nn.Conv1d(fusion_dim, d_model * 4, kernel_size=1)
        self.fusion_conv2 = nn.Conv1d(d_model * 4, d_model * 2, kernel_size=1)
        self.fusion_conv3 = nn.Conv1d(d_model * 2, d_model, kernel_size=1)

        # BatchNorm和Dropout
        self.bn1_1 = nn.BatchNorm1d(d_model)
        self.bn1_2 = nn.BatchNorm1d(d_model * 2)
        self.bn1_3 = nn.BatchNorm1d(d_model * 2)

        self.bn2_1 = nn.BatchNorm1d(d_model)
        self.bn2_2 = nn.BatchNorm1d(d_model * 2)
        self.bn2_3 = nn.BatchNorm1d(d_model * 2)

        self.bn_fusion1 = nn.BatchNorm1d(d_model * 4)
        self.bn_fusion2 = nn.BatchNorm1d(d_model * 2)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        # 全连接层
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model)
        self.fc3 = nn.Linear(d_model, d_model // 2)
        self.out_layer = nn.Linear(d_model // 2, out_num)

    def forward(self, x):
        # x: [B, 2, input_dim]

        # 分别处理两列
        col1 = x[:, 0:1, :]  # [B, 1, input_dim]
        col2 = x[:, 1:2, :]  # [B, 1, input_dim]

        # 投影到高维特征空间
        col1_emb = self.col1_proj(col1)  # [B, 1, d_model]
        col2_emb = self.col2_proj(col2)  # [B, 1, d_model]

        # 转换为CNN输入格式 [B, C, L]
        col1_conv = col1_emb.transpose(1, 2)  # [B, d_model, 1]
        col2_conv = col2_emb.transpose(1, 2)  # [B, d_model, 1]

        # 第一列多尺度卷积
        col1_feat1 = self.relu(self.bn1_1(self.col1_conv1(col1_conv)))  # [B, d_model, 1]
        col1_feat2 = self.relu(self.bn1_2(self.col1_conv2(col1_conv)))  # [B, d_model*2, 1]
        col1_feat3 = self.relu(self.bn1_3(self.col1_conv3(col1_conv)))  # [B, d_model*2, 1]
        col1_features = torch.cat([col1_feat1, col1_feat2, col1_feat3], dim=1)  # [B, d_model*5, 1]

        # 第二列多尺度卷积
        col2_feat1 = self.relu(self.bn2_1(self.col2_conv1(col2_conv)))  # [B, d_model, 1]
        col2_feat2 = self.relu(self.bn2_2(self.col2_conv2(col2_conv)))  # [B, d_model*2, 1]
        col2_feat3 = self.relu(self.bn2_3(self.col2_conv3(col2_conv)))  # [B, d_model*2, 1]
        col2_features = torch.cat([col2_feat1, col2_feat2, col2_feat3], dim=1)  # [B, d_model*5, 1]

        # 特征融合 - 这里维度是 d_model*10
        fused_features = torch.cat([col1_features, col2_features], dim=1)  # [B, d_model*10, 1]

        # 融合卷积 - 修复后的维度匹配
        fused = self.relu(self.bn_fusion1(self.fusion_conv1(fused_features)))  # [B, d_model*4, 1]
        fused = self.relu(self.bn_fusion2(self.fusion_conv2(fused)))           # [B, d_model*2, 1]
        fused = self.fusion_conv3(fused)  # [B, d_model, 1]

        # 转换为全连接层输入
        x = fused.squeeze(-1)  # [B, d_model]

        # 全连接层with残差连接
        x_input = x
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = x + x_input  # 残差连接

        x = self.relu(self.fc3(x))
        x = self.out_layer(x)  # [B, out_num]

        return x


class LSTMModel(nn.Module):
    def __init__(self, input_dim=128, d_model=256, out_num=2, num_layers=3):
        super().__init__()

        # 分别为两列构建编码器
        self.col1_proj = nn.Linear(input_dim, d_model)
        self.col2_proj = nn.Linear(input_dim, d_model)

        # 双向LSTM层
        self.col1_lstm1 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                                  bidirectional=True, batch_first=True, dropout=0.1)
        self.col1_lstm2 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                                  bidirectional=True, batch_first=True, dropout=0.1)

        self.col2_lstm1 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                                  bidirectional=True, batch_first=True, dropout=0.1)
        self.col2_lstm2 = nn.LSTM(d_model, d_model // 2, num_layers=num_layers,
                                  bidirectional=True, batch_first=True, dropout=0.1)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 特征融合方式选择
        self.fusion_method = 'weighted'  # 'mean', 'weighted', 'concat'

        if self.fusion_method == 'weighted':
            self.fusion_weights = nn.Linear(d_model * 2, 2)  # 学习权重
        elif self.fusion_method == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)

        # 全连接输出层
        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model * 4)
        self.fc3 = nn.Linear(d_model * 4, d_model)
        self.fc4 = nn.Linear(d_model, d_model // 2)
        self.out_layer = nn.Linear(d_model // 2, out_num)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 2, input_dim]

        # 分别处理两列
        col1 = x[:, 0:1, :]  # [B, 1, input_dim]
        col2 = x[:, 1:2, :]  # [B, 1, input_dim]

        # 投影到特征空间
        col1_emb = self.col1_proj(col1)  # [B, 1, d_model]
        col2_emb = self.col2_proj(col2)  # [B, 1, d_model]

        # 第一列LSTM编码
        col1_lstm1_out, _ = self.col1_lstm1(col1_emb)  # [B, 1, d_model]
        col1_lstm1_out = self.norm1(col1_lstm1_out)

        col1_lstm2_out, _ = self.col1_lstm2(col1_lstm1_out)  # [B, 1, d_model]
        col1_encoded = col1_emb + col1_lstm2_out  # 残差连接
        col1_encoded = self.norm1(col1_encoded)

        # 第二列LSTM编码
        col2_lstm1_out, _ = self.col2_lstm1(col2_emb)  # [B, 1, d_model]
        col2_lstm1_out = self.norm2(col2_lstm1_out)

        col2_lstm2_out, _ = self.col2_lstm2(col2_lstm1_out)  # [B, 1, d_model]
        col2_encoded = col2_emb + col2_lstm2_out  # 残差连接
        col2_encoded = self.norm2(col2_encoded)

        # 简单融合策略
        if self.fusion_method == 'mean':
            # 简单平均
            fused = (col1_encoded + col2_encoded) / 2  # [B, 1, d_model]

        elif self.fusion_method == 'weighted':
            # 学习权重融合
            concat_features = torch.cat([col1_encoded, col2_encoded], dim=-1)  # [B, 1, d_model*2]
            weights = torch.softmax(self.fusion_weights(concat_features), dim=-1)  # [B, 1, 2]

            w1 = weights[:, :, 0:1]  # [B, 1, 1]
            w2 = weights[:, :, 1:2]  # [B, 1, 1]
            fused = w1 * col1_encoded + w2 * col2_encoded  # [B, 1, d_model]

        else:  # concat
            # 连接后投影
            concat_features = torch.cat([col1_encoded, col2_encoded], dim=-1)  # [B, 1, d_model*2]
            fused = self.fusion_proj(concat_features)  # [B, 1, d_model]

        # 聚合维度
        x = fused.squeeze(1)  # [B, d_model]

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


class RNNModel(nn.Module):
    def __init__(self, input_dim=128, d_model=256, out_num=2, num_layers=3):
        super().__init__()

        # 分别为两列构建编码器
        self.col1_proj = nn.Linear(input_dim, d_model)
        self.col2_proj = nn.Linear(input_dim, d_model)

        # 多层RNN编码器
        self.col1_rnn1 = nn.RNN(d_model, d_model, num_layers=num_layers,
                                batch_first=True, dropout=0.1, nonlinearity='tanh')
        self.col1_rnn2 = nn.RNN(d_model, d_model, num_layers=num_layers,
                                batch_first=True, dropout=0.1, nonlinearity='relu')

        self.col2_rnn1 = nn.RNN(d_model, d_model, num_layers=num_layers,
                                batch_first=True, dropout=0.1, nonlinearity='tanh')
        self.col2_rnn2 = nn.RNN(d_model, d_model, num_layers=num_layers,
                                batch_first=True, dropout=0.1, nonlinearity='relu')

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 特征融合方式
        self.fusion_method = 'gated'  # 'mean', 'weighted', 'concat', 'gated'

        if self.fusion_method == 'weighted':
            self.fusion_weights = nn.Linear(d_model * 2, 2)
        elif self.fusion_method == 'concat':
            self.fusion_proj = nn.Linear(d_model * 2, d_model)
        elif self.fusion_method == 'gated':
            # 门控融合机制
            self.gate_fc = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )
            self.fusion_fc = nn.Linear(d_model * 2, d_model)

        # 深度全连接网络
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 2, d_model * 4),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(d_model * 2, d_model),
            nn.ReLU()
        )

        # 残差连接投影层
        self.residual_proj = nn.Linear(d_model, d_model)

        # 输出层
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(d_model // 2, out_num)
        )

        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, 2, input_dim]

        # 分别处理两列
        col1 = x[:, 0:1, :]  # [B, 1, input_dim]
        col2 = x[:, 1:2, :]  # [B, 1, input_dim]

        # 投影到特征空间
        col1_emb = self.col1_proj(col1)  # [B, 1, d_model]
        col2_emb = self.col2_proj(col2)  # [B, 1, d_model]

        # 第一列双层RNN编码
        col1_rnn1_out, _ = self.col1_rnn1(col1_emb)  # [B, 1, d_model]
        col1_rnn1_out = self.norm1(col1_rnn1_out)
        col1_rnn1_out = self.dropout(col1_rnn1_out)

        col1_rnn2_out, _ = self.col1_rnn2(col1_rnn1_out)  # [B, 1, d_model]
        col1_encoded = col1_emb + col1_rnn2_out  # 残差连接
        col1_encoded = self.norm1(col1_encoded)

        # 第二列双层RNN编码
        col2_rnn1_out, _ = self.col2_rnn1(col2_emb)  # [B, 1, d_model]
        col2_rnn1_out = self.norm2(col2_rnn1_out)
        col2_rnn1_out = self.dropout(col2_rnn1_out)

        col2_rnn2_out, _ = self.col2_rnn2(col2_rnn1_out)  # [B, 1, d_model]
        col2_encoded = col2_emb + col2_rnn2_out  # 残差连接
        col2_encoded = self.norm2(col2_encoded)

        # 特征融合策略
        if self.fusion_method == 'mean':
            # 简单平均
            fused = (col1_encoded + col2_encoded) / 2

        elif self.fusion_method == 'weighted':
            # 学习权重融合
            concat_features = torch.cat([col1_encoded, col2_encoded], dim=-1)
            weights = torch.softmax(self.fusion_weights(concat_features), dim=-1)
            w1 = weights[:, :, 0:1]
            w2 = weights[:, :, 1:2]
            fused = w1 * col1_encoded + w2 * col2_encoded

        elif self.fusion_method == 'concat':
            # 连接后投影
            concat_features = torch.cat([col1_encoded, col2_encoded], dim=-1)
            fused = self.fusion_proj(concat_features)

        else:  # gated fusion
            # 门控融合机制
            concat_features = torch.cat([col1_encoded, col2_encoded], dim=-1)  # [B, 1, d_model*2]

            gate = self.gate_fc(concat_features)  # [B, 1, d_model] - sigmoid激活
            fusion_raw = self.fusion_fc(concat_features)  # [B, 1, d_model]

            # 门控组合：gate控制新特征和原始特征的权重
            fused = gate * fusion_raw + (1 - gate) * col1_encoded

        # 聚合维度
        x = fused.squeeze(1)  # [B, d_model]

        # 深度全连接处理
        x_residual = self.residual_proj(x)  # 为残差连接准备
        x = self.fc_layers(x)  # 深度变换
        x = x + x_residual  # 残差连接

        # 输出预测
        x = self.output_layers(x)  # [B, out_num]

        return x


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, input_dim=128, d_model=256, nhead=8, num_layers=6, out_num=2):
        super().__init__()

        # 分别为两列构建投影层
        self.col1_proj = nn.Linear(input_dim, d_model)
        self.col2_proj = nn.Linear(input_dim, d_model)

        # 位置编码（虽然只有2个位置，但仍然有用）
        self.pos_encoding = PositionalEncoding(d_model, max_len=2)

        # Multi-Head Self-Attention Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 标准的4倍扩展
            dropout=0.1,
            activation='relu',
            batch_first=True,
            norm_first=True  # Pre-norm结构
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )

        # 序列聚合策略
        self.aggregation_method = 'attention_pooling'  # 'mean', 'max', 'attention_pooling', 'cls_token'

        if self.aggregation_method == 'attention_pooling':
            self.attention_pool = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=0.1,
                batch_first=True
            )
            # 可学习的查询向量
            self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))

        elif self.aggregation_method == 'cls_token':
            # 类似BERT的CLS token
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # 深度全连接网络
        self.fc_layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),  # 使用GELU激活函数（Transformer标准）
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

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化模型参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        # x: [B, 2, input_dim]
        batch_size = x.size(0)

        # 分别处理两列
        col1 = x[:, 0, :]  # [B, input_dim]
        col2 = x[:, 1, :]  # [B, input_dim]

        # 投影到Transformer维度
        col1_emb = self.col1_proj(col1)  # [B, d_model]
        col2_emb = self.col2_proj(col2)  # [B, d_model]

        # 构建序列 [B, 2, d_model]
        if self.aggregation_method == 'cls_token':
            # 添加CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, d_model]
            sequence = torch.stack([col1_emb, col2_emb], dim=1)  # [B, 2, d_model]
            sequence = torch.cat([cls_tokens, sequence], dim=1)  # [B, 3, d_model]
        else:
            sequence = torch.stack([col1_emb, col2_emb], dim=1)  # [B, 2, d_model]

        # 添加位置编码
        sequence = self.pos_encoding(sequence)

        # Transformer Encoder处理
        encoded = self.transformer_encoder(sequence)  # [B, seq_len, d_model]

        # 序列聚合
        if self.aggregation_method == 'mean':
            # 简单平均
            aggregated = encoded.mean(dim=1)  # [B, d_model]

        elif self.aggregation_method == 'max':
            # 最大池化
            aggregated, _ = encoded.max(dim=1)  # [B, d_model]

        elif self.aggregation_method == 'attention_pooling':
            # 注意力池化
            query = self.pooling_query.expand(batch_size, -1, -1)  # [B, 1, d_model]
            pooled, _ = self.attention_pool(
                query=query,
                key=encoded,
                value=encoded
            )  # [B, 1, d_model]
            aggregated = pooled.squeeze(1)  # [B, d_model]

        else:  # cls_token
            # 使用第一个位置（CLS token）的输出
            aggregated = encoded[:, 0, :]  # [B, d_model]

        # 深度全连接处理
        x_residual = self.residual_proj(aggregated)
        x = self.fc_layers(aggregated)
        x = x + x_residual  # 残差连接

        # 输出预测
        output = self.output_layers(x)  # [B, out_num]

        return output


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