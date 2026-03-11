import numpy as np
import torch
import os
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from model import FADE, FDE, FE, FE_woP
from other_model import BiLSTMModel, EncoderOnlyTransformer, UNet
from process import Process
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ViTacDataset(Dataset):
    def __init__(self, data_dir, label_dir, processor, augment=False, window_size=128, step_size=1):
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.data_files = sorted(self.data_dir.glob("*.txt"), key=lambda p: p.name)
        self.label_files = sorted(self.label_dir.glob("*.txt"), key=lambda p: p.name)
        self.processor = processor
        self.augment = augment
        self.window_size = window_size
        self.step_size = step_size
        self.data, self.labels = self.process_data()

    @staticmethod
    def _split_numbers(line: str):
        return [p for p in re.split(r"[,\s]+", line.strip()) if p]

    def add_noise(self, data):
        noise = np.random.normal(0, 0.01, size=data.shape)
        return data + noise

    def process_data(self):
        data, labels = [], []

        for data_file, label_file in zip(self.data_files, self.label_files):
            # --- data.txt: 3 列 ---
            file_data = []
            with open(data_file, "r") as f:
                for ln, line in enumerate(f, 1):
                    parts = self._split_numbers(line)
                    if len(parts) < 3:
                        continue
                    try:
                        # 3 列数据
                        vals = [float(parts[0]), float(parts[1]), float(parts[2])]
                    except ValueError:
                        continue
                    file_data.append(vals)
            print(file_data[0:5])

            # --- label.txt: 4 列 ---
            file_labels = []
            with open(label_file, "r") as f:
                for ln, line in enumerate(f, 1):
                    parts = self._split_numbers(line)
                    if len(parts) < 4:
                        raise ValueError(
                            f"标签列不足(期望≥4)，文件 `{label_file}`, 行 {ln}: {line.strip()}"
                        )
                    try:
                        # 4 列标签
                        vals = [
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        ]
                    except ValueError:
                        raise ValueError(
                            f"标签解析失败，文件 `{label_file}`, 行 {ln}: {line.strip()}"
                        )
                    file_labels.append(vals)
            print(file_labels[0:5])
            print(f"数据长度: {len(file_data)}, 标签长度: {len(file_labels)}")

            W, S = self.window_size, self.step_size
            points_per_label = W  # 你说的: 200 个数据点对应 1 个标签

            num_data = len(file_data)
            num_labels = len(file_labels)

            # 从第 0 点开始滑窗，到最后一个能完整容纳 W 点的位置
            for i in range(0, num_data - W + 1, S):
                # 当前窗口覆盖的数据下标 [i, i+W)
                # 对应到标签下标范围
                label_start = i // points_per_label
                label_end = (i + W - 1) // points_per_label + 1  # 右开区间

                # 边界保护：如果超出标签长度，就停止
                if label_start >= num_labels:
                    break
                label_end = min(label_end, num_labels)

                # 取 data 窗口
                window = file_data[i: i + W]  # [W, 3]

                # 取覆盖到的标签并求均值 -> [4]
                label_window = np.array(
                    file_labels[label_start:label_end],
                    dtype=np.float32
                )  # [K, 4]
                label_mean = label_window.mean(axis=0)

                # 3 列数据
                col1 = np.array([row[0] for row in window], dtype=np.float32)
                col2 = np.array([row[1] for row in window], dtype=np.float32)
                col3 = np.array([row[2] for row in window], dtype=np.float32)

                if self.augment:
                    col1 = self.add_noise(col1)
                    col2 = self.add_noise(col2)
                    col3 = self.add_noise(col3)

                # window_data: [3, L]
                window_data = np.stack([col1, col2, col3], axis=0).astype(np.float32)
                feature_array = self.processor.build_features(window_data)
                data.append(feature_array)
                labels.append(label_mean.astype(np.float32))

        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.ndim == 1:
            if labels.size % 4 != 0:
                raise ValueError(
                    f"labels 形状无效: {labels.shape}，请检查 `label.txt` 是否每行四列。"
                )
            labels = labels.reshape(-1, 4)

        print(data[0])
        print(labels[0])

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()               # [C, L]
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)  # [4]
        return sample, label


def plot_loss_curves(train_losses, val_losses, out_path, title="5-Fold Mean Loss Curves"):
    def _to_2d(x):
        if x is None:
            return None
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return None
        if arr.ndim == 1:
            # 兼容单折输入: [n_epochs] -> [1, n_epochs]
            arr = arr.reshape(1, -1)
        return arr

    tr = _to_2d(train_losses)
    va = _to_2d(val_losses)

    if tr is None and va is None:
        print("无 loss 记录，跳过绘图。")
        return

    # 对齐到最短 epoch
    lengths = []
    if tr is not None:
        lengths.append(tr.shape[1])
    if va is not None:
        lengths.append(va.shape[1])
    min_len = int(min(lengths)) if lengths else 0
    if min_len <= 0:
        print("loss 长度异常，跳过绘图。")
        return

    if tr is not None:
        tr = tr[:, :min_len]
        tr_mean = tr.mean(axis=0)
    else:
        tr_mean = None

    if va is not None:
        va = va[:, :min_len]
        va_mean = va.mean(axis=0)
    else:
        va_mean = None

    epochs = np.arange(1, min_len + 1)

    plt.figure(figsize=(7, 4))
    if tr_mean is not None:
        plt.plot(epochs, tr_mean, label="train_loss_mean(5fold)", linewidth=1.8)
    if va_mean is not None:
        plt.plot(epochs, va_mean, label="val_loss_mean(5fold)", linewidth=1.8)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, linewidth=0.3, alpha=0.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


def train_vitac(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs=1,
    save_path="./model/bp_artifact_best.pth",
    curve_path=None,
    curve_title=None,
    log_txt_path=None,
    log_every=10,
):
    device = next(model.parameters()).device
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 准备 loss 记录文件
    if log_txt_path is not None:
        os.makedirs(os.path.dirname(log_txt_path), exist_ok=True)
        # 覆盖写入表头
        with open(log_txt_path, "w", encoding="utf-8") as f:
            f.write("epoch\ttrain_loss\tval_loss\n")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / max(1, len(train_loader.dataset))
        train_losses.append(float(train_loss))

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_running += loss.item() * x.size(0)

        val_loss = val_running / max(1, len(val_loader.dataset))
        val_losses.append(float(val_loss))

        # 按间隔写入 txt（epoch 从 1 开始计数）
        epoch_1 = epoch + 1
        if log_txt_path is not None and (epoch_1 % log_every == 0 or epoch_1 == 1 or epoch_1 == num_epochs):
            with open(log_txt_path, "a", encoding="utf-8") as f:
                f.write(f"{epoch_1}\t{train_loss:.6f}\t{val_loss:.6f}\n")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch_1}] val_loss 改进为 {val_loss:.6f}，已保存 `{save_path}`")

        print(f"Epoch {epoch_1}/{num_epochs} | train_loss={tf(train_loss)} | val_loss={tf(val_loss)}")

    if curve_path is not None:
        plot_loss_curves(
            train_losses,
            val_losses,
            curve_path,
            title=curve_title or "train/val loss",
        )

    return best_val_loss, train_losses, val_losses


def tf(x):
    return f"{x:.6f}"


if __name__ == "__main__":
    data_dir = "./data/static"
    label_dir = "./label/static"
    base_model_dir = "./model"
    os.makedirs(base_model_dir, exist_ok=True)

    out_dir = "./output"
    os.makedirs(out_dir, exist_ok=True)

    augment = True
    window_size = 200
    batch_size = 16
    num_epochs = 150
    seed = 42

    processor = Process(model_path=None)
    dataset = ViTacDataset(
        data_dir,
        label_dir,
        processor=processor,
        augment=augment,
        window_size=window_size,
        step_size=10,
    )

    N = len(dataset)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    test_len = N - train_len - val_len
    train_base, val_fixed, test_fixed = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed),
    )

    print(f"使用设备: {device}")
    print(f"数据集大小: {N}")
    print(f"训练基集: {len(train_base)}, 验证集: {len(val_fixed)}, 测试集: {len(test_fixed)}")

    kf = KFold(n_splits=2, shuffle=True, random_state=seed)
    fold_best_losses = []
    fold_paths = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(train_base))), start=1):
        train_fold = Subset(train_base, tr_idx)
        val_fold = Subset(train_base, va_idx)

        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)

        ''' 模型选择 '''
        model = FADE(input_len=window_size).to(device)
        # model = FDE(input_len=window_size).to(device)
        # model = FE(input_len=window_size).to(device)
        # model = FE_woP(input_len=window_size).to(device)

        # model = MultiResUNet1D(input_len=window_size).to(device)
        # model = BiLSTMModel(input_len=window_size).to(device)
        # model = EncoderOnlyTransformer(input_len=window_size).to(device)
        # model = UNet(input_len=window_size).to(device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        save_path = os.path.join(base_model_dir, f"bp_cv_fold_{fold}.pth")
        curve_path = os.path.join(out_dir, f"loss_curve_fold_{fold}.png")
        log_txt_path = os.path.join(out_dir, f"loss_log_fold_{fold}.txt")

        best_loss, train_losses, val_losses = train_vitac(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            num_epochs=num_epochs,
            save_path=save_path,
            curve_path=curve_path,
            curve_title=f"Fold {fold} loss curves",
            log_txt_path=log_txt_path,
            log_every=10,
        )

        fold_best_losses.append(best_loss)
        fold_paths.append(save_path)
        print(f"[Fold {fold}] 最优验证损失: {tf(best_loss)} | 权重文件: `{save_path}`")

