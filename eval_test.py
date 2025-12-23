import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import Trans, TransNoPressure, TransNoPressDifAttn
from process import Process

from train import ViTacDataset


def collect_predictions(model, loader, device):
    model.eval()
    ref, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()  # [B, 4]
            ref.append(y.numpy())           # [B, 4]
            preds.append(pred)
    if not ref:
        return np.empty((0, 4), dtype=np.float32), np.empty((0, 4), dtype=np.float32)
    y_true = np.concatenate(ref, axis=0)
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/static")
    parser.add_argument("--label_dir", type=str, default="./label/static")
    parser.add_argument("--best_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="./output")
    parser.add_argument("--window_size", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--step_size", type=int, default=10)
    args = parser.parse_args()



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    processor = Process(model_path=None)
    dataset = ViTacDataset(
        args.data_dir,
        args.label_dir,
        processor=processor,
        augment=args.augment,      # 测试建议不增强；如需保持一致可传 --augment
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # 与 train.py 一致的 6/2/2 切分方式，保证 test_fixed 可复现
    N = len(dataset)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    test_len = N - train_len - val_len
    _, _, test_fixed = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    test_loader = DataLoader(test_fixed, batch_size=args.batch_size, shuffle=False)

    ''' 模型选择'''
    # model = Trans(input_len=args.window_size).to(device)
    model = TransNoPressure(input_len=args.window_size).to(device)
    # model = TransNoPressDifAttn(input_len=args.window_size).to(device)

    state = torch.load(args.best_path, map_location=device)
    model.load_state_dict(state)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    # Bland-Altman
    plot_bland_altman(y_true[:, 0], y_pred[:, 0], os.path.join(args.out_dir, "bland_altman_sbp.png"), title="SBP (mmHg)")
    plot_bland_altman(y_true[:, 1], y_pred[:, 1], os.path.join(args.out_dir, "bland_altman_dbp.png"), title="DBP (mmHg)")
    plot_bland_altman(y_true[:, 2], y_pred[:, 2], os.path.join(args.out_dir, "bland_altman_spo2.png"), title="SpO2 (%)")
    plot_bland_altman(y_true[:, 3], y_pred[:, 3], os.path.join(args.out_dir, "bland_altman_hr.png"), title="HR (bpm)")

    # Correlation
    plot_correlation(y_true[:, 0], y_pred[:, 0], os.path.join(args.out_dir, "corr_sbp.png"), title="SBP correlation")
    plot_correlation(y_true[:, 1], y_pred[:, 1], os.path.join(args.out_dir, "corr_dbp.png"), title="DBP correlation")
    plot_correlation(y_true[:, 2], y_pred[:, 2], os.path.join(args.out_dir, "corr_spo2.png"), title="SpO2 correlation")
    plot_correlation(y_true[:, 3], y_pred[:, 3], os.path.join(args.out_dir, "corr_hr.png"), title="HR correlation")

    # 保存误差
    errors = (y_true - y_pred).astype(np.float32)  # [N, 4]
    out_err_path = os.path.join(args.out_dir, "out_error.txt")
    np.savetxt(
        out_err_path,
        errors,
        fmt="%.6f",
        delimiter="\t",
        header="SBP_err\tDBP_err\tSpO2_err\tHR_err",
    )
    print(f"已保存: {out_err_path}")


if __name__ == "__main__":
    main()
