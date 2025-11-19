import numpy as np
import torch
import os
import re
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from model import Trans
from process import Process
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class VADDataset(Dataset):
    def __init__(self, data_dir, label_dir, processor, augment=False, window_size=128, step_size=1):
        from pathlib import Path
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.data_files = sorted(self.data_dir.glob("data.txt"))
        self.label_files = sorted(self.label_dir.glob("label.txt"))
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
            file_data = []
            with open(data_file, "r") as f:
                for ln, line in enumerate(f, 1):
                    parts = self._split_numbers(line)
                    if len(parts) < 2:
                        continue
                    try:
                        vals = [float(parts[0]), float(parts[1])]
                    except ValueError:
                        continue
                    file_data.append(vals)

            file_labels = []
            with open(label_file, "r") as f:
                for ln, line in enumerate(f, 1):
                    parts = self._split_numbers(line)
                    if len(parts) < 3:
                        raise ValueError(f"标签列不足(期望≥3)，文件 `{label_file}`, 行 {ln}: {line.strip()}")
                    try:
                        vals = [float(parts[0]), float(parts[1]), float(parts[2])]
                    except ValueError:
                        raise ValueError(f"标签解析失败，文件 `{label_file}`, 行 {ln}: {line.strip()}")
                    file_labels.append(vals)

            W, S = self.window_size, self.step_size
            Ld, Ll = int(len(file_data)/3), int(len(file_labels)/3)

            for i in range(Ld, 3*Ld - W + 1, S):
                if i + W > 3*Ll:
                    break

                window = file_data[i:i + W]
                label_window = np.array(file_labels[i:i + W], dtype=np.float32)  # [W, 3]
                label_mean = label_window.mean(axis=0)                           # [3]

                col1 = np.array([row[0] for row in window], dtype=np.float32)
                col2 = np.array([row[1] for row in window], dtype=np.float32)

                if self.augment:
                    col1 = self.add_noise(col1)
                    col2 = self.add_noise(col2)

                window_data = np.stack([col1, col2], axis=0).astype(np.float32)  # [2, L]
                feature_array = self.processor.build_features(window_data)        # [4, L]
                data.append(feature_array)
                labels.append(label_mean.astype(np.float32))

        data = np.asarray(data, dtype=np.float32)                                # [N, 4, L]
        labels = np.asarray(labels, dtype=np.float32)                            # [N, 3] or [N*3]
        if labels.ndim == 1:
            if labels.size % 3 != 0:
                raise ValueError(f"labels 形状无效: {labels.shape}，请检查 `label.txt` 是否每行三列 SBP,DBP,HR。")
            labels = labels.reshape(-1, 3)

        return data, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()         # [4, L]
        label = torch.as_tensor(self.labels[idx], dtype=torch.float32)  # [3]
        return sample, label


def collect_predictions(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()  # [B, 3]
            ys.append(y.numpy())           # [B, 3]
            preds.append(pred)
    if not ys:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.float32)
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return y_true, y_pred


def plot_bland_altman(y_true_1d, y_pred_1d, out_path, title="Bland-Altman"):
    y_true_1d = np.asarray(y_true_1d).ravel()
    y_pred_1d = np.asarray(y_pred_1d).ravel()
    if y_true_1d.size == 0:
        print("测试集为空，跳过绘图。")
        return

    mean_vals = (y_pred_1d + y_true_1d) / 2.0
    diff = y_pred_1d - y_true_1d
    md = np.mean(diff)
    sd = np.std(diff, ddof=1) if diff.size > 1 else 0.0
    loa_upper = md + 1.96 * sd
    loa_lower = md - 1.96 * sd

    plt.figure(figsize=(6, 4))
    plt.scatter(mean_vals, diff, s=10, alpha=0.6, edgecolors="none")
    plt.axhline(md, color="red", linestyle="--", label=f"Bias={md:.3f}")
    plt.axhline(loa_upper, color="gray", linestyle="--", label=f"+1.96SD={loa_upper:.3f}")
    plt.axhline(loa_lower, color="gray", linestyle="--", label=f"-1.96SD={loa_lower:.3f}")
    plt.xlabel("Mean of prediction and reference")
    plt.ylabel("Prediction - reference")
    plt.title(title)
    plt.legend(loc="best")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


def train_vad(model, train_loader, val_loader, criterion, optimizer, num_epochs=1, save_path='./model/bp_artifact_best.pth'):
    device = next(model.parameters()).device
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)            # [B, 4, L]
            y = y.to(device)            # [B, 3]

            optimizer.zero_grad()
            pred = model(x)             # [B, 3]
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_running = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_running += loss.item() * x.size(0)

        val_count = max(1, len(val_loader.dataset))
        val_loss = val_running / val_count

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"[Epoch {epoch + 1}] val_loss 改进为 {val_loss:.6f}，已保存 `{save_path}`")

        print(f"Epoch {epoch + 1}/{num_epochs} | train_loss={tf(train_loss)} | val_loss={tf(val_loss)}")

    return best_val_loss


def tf(x):
    return f"{x:.6f}"


def plot_correlation(y_true_1d, y_pred_1d, out_path, title="Correlation"):
    y_true_1d = np.asarray(y_true_1d).ravel()
    y_pred_1d = np.asarray(y_pred_1d).ravel()
    if y_true_1d.size == 0:
        print("测试集为空，跳过绘图。")
        return

    r = float(np.corrcoef(y_true_1d, y_pred_1d)[0, 1]) if y_true_1d.size > 1 else 0.0
    k, b = np.polyfit(y_true_1d, y_pred_1d, 1) if y_true_1d.size > 1 else (1.0, 0.0)

    x_min, x_max = np.min(y_true_1d), np.max(y_true_1d)
    xs = np.linspace(x_min, x_max, 100)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_true_1d, y_pred_1d, s=10, alpha=0.6, edgecolors="none")
    plt.plot([x_min, x_max], [x_min, x_max], "k--", linewidth=1, label="y=x")
    plt.plot(xs, k * xs + b, "r-", linewidth=1, label=f"fit: y={k:.3f}x+{b:.3f}")
    plt.xlabel("Reference")
    plt.ylabel("Prediction")
    plt.title(f"{title} (r={r:.3f}, R^2={r*r:.3f})")
    plt.legend(loc="best")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


if __name__ == "__main__":
    data_dir = "./dataset/0826"
    label_dir = "./dataset/0826"
    base_model_dir = "./model"
    os.makedirs(base_model_dir, exist_ok=True)

    augment = True
    window_size = 128
    batch_size = 16
    num_epochs = 100
    seed = 42

    processor = Process(model_path=None)
    dataset = VADDataset(data_dir, label_dir, processor=processor,
                         augment=augment, window_size=window_size, step_size=10)

    N = len(dataset)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    test_len = N - train_len - val_len
    train_base, val_fixed, test_fixed = random_split(
        dataset, [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(seed)
    )

    print(f"使用设备: {device}")
    print(f"数据集大小: {N}")
    print(f"训练基集: {len(train_base)}, 验证集: {len(val_fixed)}, 测试集: {len(test_fixed)}")
    # 交叉验证训练
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    fold_best_losses = []
    fold_paths = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(range(len(train_base))), start=1):
        train_fold = Subset(train_base, tr_idx)
        val_fold = Subset(train_base, va_idx)

        train_loader = DataLoader(train_fold, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_fold, batch_size=batch_size, shuffle=False)

        model = Trans(input_dim=window_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        save_path = os.path.join(base_model_dir, f"bp_cv_fold_{fold}.pth")
        best_loss = train_vad(model, train_loader, val_loader, criterion, optimizer,
                              num_epochs=num_epochs, save_path=save_path)

        fold_best_losses.append(best_loss)
        fold_paths.append(save_path)
        print(f"[Fold {fold}] 最优验证损失: {tf(best_loss)} | 权重文件: `{save_path}`")

    best_fold = int(np.argmin(fold_best_losses)) + 1
    best_path = fold_paths[best_fold - 1]
    print(f"选择验证损失最小的折: Fold {best_fold} -> `{best_path}`")

    # 在测试集上评估并绘制 Bland-Altman 图
    test_loader = DataLoader(test_fixed, batch_size=batch_size, shuffle=False)
    model = Trans(input_dim=window_size).to(device)

    state = torch.load(best_path, map_location=device)
    model.load_state_dict(state)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    out_dir = "./output"
    plot_bland_altman(y_true[:, 0], y_pred[:, 0],
                      os.path.join(out_dir, "bland_altman_sbp.png"),
                      title="SBP (mmHg)")
    plot_bland_altman(y_true[:, 1], y_pred[:, 1],
                      os.path.join(out_dir, "bland_altman_dbp.png"),
                      title="DBP (mmHg)")
    plot_bland_altman(y_true[:, 2], y_pred[:, 2],
                      os.path.join(out_dir, "bland_altman_hr.png"),
                      title="HR (bpm)")
    plot_correlation(y_true[:, 0], y_pred[:, 0],
                     os.path.join(out_dir, "corr_sbp.png"),
                     title="SBP correlation")
    plot_correlation(y_true[:, 1], y_pred[:, 1],
                     os.path.join(out_dir, "corr_dbp.png"),
                     title="DBP correlation")
    plot_correlation(y_true[:, 2], y_pred[:, 2],
                     os.path.join(out_dir, "corr_hr.png"),
                     title="HR correlation")
