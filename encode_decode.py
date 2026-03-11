import torch
import torch.nn as nn
import torch.nn.functional as F


class EncodeBlock(nn.Module):
    """
    双流编码：
    - 短时流：4层通道提取（核长 1,3,3,3），保持结构不变。
    - 长时流：2层通道提取（核长为短时对应层的2倍：6,6）。
    - 融合：长时每次提取后，与短时每2次提取后的特征图相加取均值：
        * 融合1：短时 conv2 输出 与 长时 conv1 输出 -> 均值 -> 作为短时 conv3 的输入
        * 融合2：短时 conv4 输出 与 长时 conv2 输出 -> 均值 -> 进入长度投影与压缩
    - 其余部分（长度投影与压缩）保持不变。
    输入:  x [B, 2, input_len]
    输出:  y [B, d_model]
    """
    def __init__(self, input_len: int, d_model: int, c_mid: int = 64):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        # 短时流通道配置：c1,c2,c3,c4(=c_feat)
        c1 = c_mid
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2  # = c_feat
        self.c_feat = c4

        # 短时流 4层通道提取（核长 1,3,3,3）
        self.conv_c1 = nn.Conv1d(in_channels=2,  out_channels=c1, kernel_size=1, bias=False)
        self.bn_c1   = nn.BatchNorm1d(c1)

        self.conv_c2 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1, bias=False)
        self.bn_c2   = nn.BatchNorm1d(c2)

        self.conv_c3 = nn.Conv1d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1, bias=False)
        self.bn_c3   = nn.BatchNorm1d(c3)

        self.conv_c4 = nn.Conv1d(in_channels=c3, out_channels=c4, kernel_size=3, padding=1, bias=False)
        self.bn_c4   = nn.BatchNorm1d(c4)

        # 长时流 2层通道提取（核长为短时对应层的2倍：6,6），输出通道与短时第2/第4层对齐以便融合
        k_long = 6  # 2 * 3
        self.long_conv1 = nn.Conv1d(in_channels=2,  out_channels=c2, kernel_size=k_long, padding=k_long // 2, bias=False)
        self.long_bn1   = nn.BatchNorm1d(c2)

        self.long_conv2 = nn.Conv1d(in_channels=c2, out_channels=c4, kernel_size=k_long, padding=k_long // 2, bias=False)
        self.long_bn2   = nn.BatchNorm1d(c4)

        # 长度维投影与压缩（保持不变）
        self.conv_l1 = nn.Conv1d(in_channels=input_len, out_channels=d_model, kernel_size=1, bias=False)
        self.bn_l1   = nn.BatchNorm1d(d_model)
        self.conv_l2 = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.c_feat, bias=True)

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _match_len(ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        将 y 在最后一维对齐到 ref 的长度（偶数卷积核会导致长度+1，这里裁剪或右侧补零对齐）
        """
        L_ref = ref.size(-1)
        L_y = y.size(-1)
        if L_y > L_ref:
            return y[..., :L_ref]
        if L_y < L_ref:
            pad = L_ref - L_y
            return F.pad(y, (0, pad), mode="constant", value=0.0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 2 and x.size(2) == self.input_len, "输入形状应为 [B, 2, input_len]"
        B, _, L = x.shape

        # 短时流前两层
        st1 = self.relu(self.bn_c1(self.conv_c1(x)))          # [B, c1, L]
        st2 = self.relu(self.bn_c2(self.conv_c2(st1)))        # [B, c2, L]

        # 长时流第一层（核长=6），与 st2 融合
        lt1 = self.relu(self.long_bn1(self.long_conv1(x)))    # [B, c2, L(+1 或 L)]
        lt1 = self._match_len(st2, lt1)                       # 对齐到 L
        fused2 = 0.5 * (st2 + lt1)                            # [B, c2, L]

        # 融合结果作为短时第三层输入，继续短时流
        st3 = self.relu(self.bn_c3(self.conv_c3(fused2)))     # [B, c3, L]
        st4 = self.relu(self.bn_c4(self.conv_c4(st3)))        # [B, c4, L]

        # 长时流第二层（核长=6），与 st4 融合
        lt2 = self.relu(self.long_bn2(self.long_conv2(lt1)))  # [B, c4, L(+1 或 L)]
        lt2 = self._match_len(st4, lt2)                       # 对齐到 L
        fused4 = 0.5 * (st4 + lt2)                            # [B, c4, L]
        '''
        # 进入原有长度投影与压缩
        y = fused4.transpose(1, 2)                            # [B, L, c4]
        y = self.relu(self.bn_l1(self.conv_l1(y)))            # [B, d_model, c4]
        y = self.conv_l2(y)                                   # [B, d_model, 1]
        y = y.transpose(1, 2)                                  # [B, 1, d_model]
        '''
        return fused4.transpose(1, 2)                            # [B, L, c4]


class DecodeBlock(nn.Module):
    """
    与当前 EncodeBlock 对称的双流解码（适配 EncodeBlock 输出: [B, L, c_feat]）:

    输入:  x [B, L, c_feat]
    输出:  y [B, 2, L]

    双流解码：
    - 短时流：4层通道还原（核长 3,3,3,1）：c4 -> c3 -> c2 -> c1 -> 2
    - 长时流：2层通道还原（核长 6,6）：c4 -> c2 -> 2（其中第 1 层与短时第 2 层融合；第 2 层与短时第 4 层融合）
    - 融合（与编码对应）：
        * 融合1：短时 deconv2 输出 与 长时 deconv1 输出 -> 均值 -> 作为短时 deconv3 的输入
        * 融合2：短时 deconv4 输出 与 长时 deconv2 输出 -> 均值 -> 输出
    """
    def __init__(self, input_len: int, d_model: int, c_mid: int = 64):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        c1 = c_mid
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2
        self.c_feat = c4

        # 短时流：c4 -> c3 -> c2 -> c1 -> 2（核长 3,3,3,1）
        self.deconv_c4 = nn.ConvTranspose1d(in_channels=c4, out_channels=c3, kernel_size=3, padding=1, bias=False)
        self.bn_c4 = nn.BatchNorm1d(c3)

        self.deconv_c3 = nn.ConvTranspose1d(in_channels=c3, out_channels=c2, kernel_size=3, padding=1, bias=False)
        self.bn_c3 = nn.BatchNorm1d(c2)

        self.deconv_c2 = nn.ConvTranspose1d(in_channels=c2, out_channels=c1, kernel_size=3, padding=1, bias=False)
        self.bn_c2 = nn.BatchNorm1d(c1)

        self.deconv_c1 = nn.ConvTranspose1d(in_channels=c1, out_channels=2, kernel_size=1, bias=True)

        # 长时流：2 层（核长为短时对应层的 2 倍：6,6）
        k_long = 6
        self.long_deconv1 = nn.ConvTranspose1d(in_channels=c4, out_channels=c2,
                                               kernel_size=k_long, padding=k_long // 2, bias=False)
        self.long_bn1 = nn.BatchNorm1d(c2)

        self.long_deconv2 = nn.ConvTranspose1d(in_channels=c2, out_channels=2,
                                               kernel_size=k_long, padding=k_long // 2, bias=True)
        # 输出通道为 2，一般不再接 BN；如需可自行加

        self.relu = nn.ReLU(inplace=True)

    @staticmethod
    def _match_len(ref: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        将 y 在最后一维对齐到 ref 的长度（偶数卷积核可能导致长度 +/-1，这里裁剪或右侧补零对齐）
        """
        L_ref = ref.size(-1)
        L_y = y.size(-1)
        if L_y > L_ref:
            return y[..., :L_ref]
        if L_y < L_ref:
            pad = L_ref - L_y
            return F.pad(y, (0, pad), mode="constant", value=0.0)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(2) == self.c_feat, r"输入形状应为 \[B, L, c\_feat\]"
        # [B, L, c_feat] -> [B, c_feat, L]
        y = x.transpose(1, 2)

        # 短时流前两层：c4 -> c3 -> c2
        st1 = self.relu(self.bn_c4(self.deconv_c4(y)))     # [B, c3, L]
        st2 = self.relu(self.bn_c3(self.deconv_c3(st1)))   # [B, c2, L]

        # 长时流第一层：c4 -> c2，并与 st2 融合
        lt1 = self.relu(self.long_bn1(self.long_deconv1(y)))  # [B, c2, L(+1 或 L)]
        lt1 = self._match_len(st2, lt1)
        fused2 = 0.5 * (st2 + lt1)                          # [B, c2, L]

        # 融合结果作为短时第三层输入：c2 -> c1
        st3 = self.relu(self.bn_c2(self.deconv_c2(fused2)))  # [B, c1, L]
        st4 = self.deconv_c1(st3)                            # [B, 2,  L]

        # 长时流第二层：c2 -> 2（输入用 lt1），与 st4 融合输出
        lt2 = self.long_deconv2(lt1)                         # [B, 2, L(+1 或 L)]
        lt2 = self._match_len(st4, lt2)
        out = 0.5 * (st4 + lt2)                              # [B, 2, L]
        return out


class EncodeBlockSTOnly(nn.Module):
    """
    仅短时流编码：
    - 只使用短时流：4层通道提取（核长 1,3,3,3）
    - 不使用长时流和融合操作
    - 保持与EncodeBlock相同的输入输出格式
    输入:  x [B, 2, input_len]
    输出:  y [B, L, c_feat] (与EncodeBlock保持一致)
    """
    def __init__(self, input_len: int, d_model: int, c_mid: int = 64):
        super().__init__()
        self.input_len = input_len
        self.d_model = d_model

        # 短时流通道配置：c1,c2,c3,c4(=c_feat)
        c1 = c_mid
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2  # = c_feat
        self.c_feat = c4

        # 短时流 4层通道提取（核长 1,3,3,3）
        self.conv_c1 = nn.Conv1d(in_channels=2,  out_channels=c1, kernel_size=1, bias=False)
        self.bn_c1   = nn.BatchNorm1d(c1)

        self.conv_c2 = nn.Conv1d(in_channels=c1, out_channels=c2, kernel_size=3, padding=1, bias=False)
        self.bn_c2   = nn.BatchNorm1d(c2)

        self.conv_c3 = nn.Conv1d(in_channels=c2, out_channels=c3, kernel_size=3, padding=1, bias=False)
        self.bn_c3   = nn.BatchNorm1d(c3)

        self.conv_c4 = nn.Conv1d(in_channels=c3, out_channels=c4, kernel_size=3, padding=1, bias=False)
        self.bn_c4   = nn.BatchNorm1d(c4)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3 and x.size(1) == 2 and x.size(2) == self.input_len, "输入形状应为 [B, 2, input_len]"
        B, _, L = x.shape

        # 仅短时流四层处理
        st1 = self.relu(self.bn_c1(self.conv_c1(x)))          # [B, c1, L]
        st2 = self.relu(self.bn_c2(self.conv_c2(st1)))        # [B, c2, L]
        st3 = self.relu(self.bn_c3(self.conv_c3(st2)))        # [B, c3, L]
        st4 = self.relu(self.bn_c4(self.conv_c4(st3)))        # [B, c4, L]

        return st4.transpose(1, 2)                             # [B, L, c4]
