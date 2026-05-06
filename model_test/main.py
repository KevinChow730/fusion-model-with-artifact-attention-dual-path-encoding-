import os
import re
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import Trans  # 如需换模型：TransNoPressure / TransNoPressDifAttn
from process import Process


def _split_numbers(line: str):
    return [p for p in re.split(r"[,\s]+", line.strip()) if p]


def load_static_data(data_path: str):
    rows = []
    with open(data_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = _split_numbers(line)
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"`{data_path}` 无有效数据行。")
    return np.asarray(rows, dtype=np.float32)  # [N, 3]


def load_static_labels(label_path: str):
    rows = []
    with open(label_path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            parts = _split_numbers(line)
            if len(parts) < 4:
                raise ValueError(f"`{label_path}` 行 {ln} 标签列不足(期望≥4)。")
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])])
            except ValueError:
                raise ValueError(f"`{label_path}` 行 {ln} 标签解析失败。")
    if not rows:
        raise ValueError(f"`{label_path}` 无有效标签行。")
    return np.asarray(rows, dtype=np.float32)  # [M, 4]


def make_windows_features(
    data_3col: np.ndarray,
    labels_4col: np.ndarray,
    processor: Process,
    window_size: int = 200,
    step_size: int = 200,
):
    """
    data_3col: [N, 3]
    labels_4col: [M, 4]
    返回:
      features_list: List[np.ndarray] each [C, L]
      refs_4d: np.ndarray [K, 4] 每个窗口对应4维ref(取覆盖标签均值)
      win_starts: np.ndarray [K]
    """
    N = int(data_3col.shape[0])
    M = int(labels_4col.shape[0])
    W, S = int(window_size), int(step_size)
    points_per_label = W  # 与 train.py 一致: W 个点对应 1 个标签

    features_list = []
    refs_list = []
    win_starts = []

    for i in range(0, N - W + 1, S):
        label_start = i // points_per_label
        label_end = (i + W - 1) // points_per_label + 1
        if label_start >= M:
            break
        label_end = min(label_end, M)

        window = data_3col[i: i + W]  # [W, 3]

        # 每个窗口的 ref: 取覆盖到的标签均值 -> [4]
        label_mean = labels_4col[label_start:label_end].mean(axis=0).astype(np.float32)  # [4]
        refs_list.append(label_mean)

        window_data = np.stack(
            [window[:, 0].astype(np.float32), window[:, 1].astype(np.float32), window[:, 2].astype(np.float32)],
            axis=0,
        )  # [3, W]
        feat = processor.build_features(window_data)  # [C, L]
        features_list.append(feat.astype(np.float32))
        win_starts.append(i)

    if not features_list:
        raise ValueError("滑窗后没有生成任何窗口，请检查数据长度/窗口大小/步长。")

    refs_4d = np.stack(refs_list, axis=0).astype(np.float32)  # [K, 4]
    return features_list, refs_4d, np.asarray(win_starts, dtype=np.int64)


def run_inference(
    weights_path: str,
    data_path: str = "./data/static/data.txt",
    label_path: str = "./data/static/label.txt",
    out_csv: str = "./output/static_pred.csv",
    out_fig: str = "./output/static_pred_vs_ref.png",
    window_size: int = 200,
    step_size: int = 200,
    ref_col_index: int = 0,  # 保留参数但不再影响 2x2 对比；需要只看某一列可用于别处
):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_fig), exist_ok=True)

    data_3col = load_static_data(data_path)
    labels_4col = load_static_labels(label_path)

    processor = Process(model_path=None)
    feats, refs_4d, win_starts = make_windows_features(
        data_3col,
        labels_4col,
        processor,
        window_size=window_size,
        step_size=step_size,
    )

    model = Trans(input_len=window_size).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    preds = []
    with torch.no_grad():
        for feat in feats:
            x = torch.from_numpy(feat).unsqueeze(0).float().to(device)  # [1, C, L]
            y = model(x).squeeze(0).detach().cpu().numpy().astype(np.float32)  # [4]
            preds.append(y)
    preds = np.stack(preds, axis=0)  # [K, 4]

    # 保存 CSV（同时保存 4 维 ref 和 4 维 pred）
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "win_index", "start",
            "ref_0", "ref_1", "ref_2", "ref_3",
            "pred_0", "pred_1", "pred_2", "pred_3",
        ])
        for k in range(preds.shape[0]):
            w.writerow(
                [k, int(win_starts[k])]
                + [float(v) for v in refs_4d[k].tolist()]
                + [float(v) for v in preds[k].tolist()]
            )

    # 画图：pred[d] vs ref[d]，2x2 子图
    t = np.arange(preds.shape[0], dtype=np.int64)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    axes = axes.reshape(-1)

    for d in range(4):
        ax = axes[d]
        ax.plot(t, refs_4d[:, d], label=f"ref[{d}](label mean)", linewidth=1.6)
        ax.plot(t, preds[:, d], label=f"pred[{d}]", linewidth=1.6)
        ax.set_title(f"Dim {d}: pred[{d}] vs ref[{d}]")
        ax.set_ylabel("Value")
        ax.grid(True, linewidth=0.3, alpha=0.6)
        ax.legend(loc="best")

    axes[2].set_xlabel("Window index")
    axes[3].set_xlabel("Window index")

    fig.suptitle("Static inference: pred vs ref (time series)", y=0.98)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=200)
    plt.close(fig)

    print(f"已保存预测CSV: `{out_csv}`")
    print(f"已保存对比图: `{out_fig}`")


if __name__ == "__main__":
    # 把这里改成你要用的权重文件（例如训练输出的某个fold）
    weights_path = "./model/bp_cv_fold_1.pth"

    run_inference(
        weights_path=weights_path,
        data_path="./data/static/data_3.txt",
        label_path="./label/static/label_3.txt",
        out_csv="./output/static_pred.csv",
        out_fig="./output/static_pred_vs_ref.png",
        window_size=200,
        step_size=200,
        ref_col_index=0,
    )
