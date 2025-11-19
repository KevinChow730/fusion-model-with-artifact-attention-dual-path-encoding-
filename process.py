import numpy as np
import torch
from model import Trans


class Process:
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.__load_model(model_path) if model_path else None

    def __load_model(self, model_path: str):
        # 加载 state_dict
        state = torch.load(model_path, map_location=self.device, weights_only=True)

        # 从权重中推断 d_model 与 input_dim
        # enc_a.conv_l1.weight 的形状应为 [d_model, input_dim, 1]
        w = state["enc_a.conv_l1.weight"]
        d_model, input_dim = int(w.shape[0]), int(w.shape[1])

        # 实例化与 checkpoint 完全匹配的模型
        model = Trans(input_dim=input_dim, d_model=d_model).to(self.device)

        # 严格加载
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    def build_features(self, window_data: np.ndarray) -> np.ndarray:
        """
        将两列时间序列转换为特征: 通道顺序 `[a_time, a_freq, b_time, b_freq]`，形状为 `[4, L]`。
        参数:
            window_data: np.ndarray，形状 `[2, L]`
        返回:
            np.ndarray，形状 `[4, L]`，dtype=float32
        """
        if window_data.ndim != 2 or window_data.shape[0] != 2:
            raise ValueError("window_data 形状必须为 `[2, L]`")
        time = window_data.astype(np.float32)  # [2, L]
        L = time.shape[1]
        fft_mag = (np.abs(np.fft.fft(time, axis=-1)) / float(L)).astype(np.float32)  # [2, L]
        chw = np.stack([time[0], fft_mag[0], time[1], fft_mag[1]], axis=0).astype(np.float32)  # [4, L]
        return chw

    def process_window(self, window_data: np.ndarray):
        """
        推理入口：将`window_data`转换为 `[1, 4, L]` 后送入模型，返回预测结果。
        """
        if self.model is None:
            raise RuntimeError("未加载模型。实例化 `process` 时请提供 `model_path`。")
        chw = self.build_features(window_data)                     # [4, L] (np)
        input_tensor = torch.from_numpy(chw).unsqueeze(0).float()  # [1, 4, L]
        input_tensor = input_tensor.to(self.device)

        self.model.eval()
        with torch.no_grad():
            output = self.model(input_tensor).squeeze(0).cpu().numpy()
            # 如需与先前逻辑一致，这里可选择性做偏移；训练阶段不建议做该操作
            # output[0] = output[0] - 18
        return output