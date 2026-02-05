import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import Trans, TransNoDiffAttn, TransNoPressDifAttn, TransNoDiffAttnSTOnly
from other_model import BiLSTMModel, EncoderOnlyTransformer, UNet, MultiResUNet1D
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


def _safe_import_scipy_stats():
    try:
        from scipy import stats  # type: ignore
        return stats
    except Exception as e:
        print("缺少 scipy，无法进行假设检验。请先安装：pip install scipy")
        raise e


def collect_labels(subset):
    # subset: torch.utils.data.Subset(ViTacDataset)
    ys = []
    for i in range(len(subset)):
        _, y = subset[i]
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        y = np.asarray(y).reshape(-1)
        ys.append(y)
    if not ys:
        return np.empty((0, 4), dtype=np.float32)
    y_all = np.stack(ys, axis=0).astype(np.float32)  # [N, 4]
    return y_all


def compute_label_stats(y_all):
    # y_all: [N, 4]
    stats_dict = {}
    if y_all.size == 0:
        return stats_dict

    for j in range(y_all.shape[1]):
        v = y_all[:, j].astype(np.float64)
        q1 = float(np.percentile(v, 25))
        q3 = float(np.percentile(v, 75))
        stats_dict[j] = {
            "n": int(v.size),
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "median": float(np.median(v)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "q1": q1,
            "q3": q3,
            "iqr": float(q3 - q1),
        }
    return stats_dict


def save_stats_table(out_path, stats_train, stats_val, stats_test, label_names):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _row(split_name, j, d):
        return [
            split_name,
            label_names[j],
            str(d["n"]),
            f'{d["mean"]:.6f}',
            f'{d["std"]:.6f}',
            f'{d["median"]:.6f}',
            f'{d["q1"]:.6f}',
            f'{d["q3"]:.6f}',
            f'{d["iqr"]:.6f}',
            f'{d["min"]:.6f}',
            f'{d["max"]:.6f}',
        ]

    header = [
        "split",
        "label",
        "n",
        "mean",
        "std",
        "median",
        "q1",
        "q3",
        "iqr",
        "min",
        "max",
    ]

    lines = ["\t".join(header)]
    for j in range(len(label_names)):
        for split_name, dct in [("train", stats_train), ("val", stats_val), ("test", stats_test)]:
            d = dct.get(j, None)
            if d is None:
                continue
            lines.append("\t".join(_row(split_name, j, d)))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存: {out_path}")


def distribution_tests(y_train, y_val, y_test, out_path, label_names):
    # 非参数检验：Kruskal-Wallis（三组），Mann-Whitney U（两两）+ Bonferroni，KS（两两）
    stats = _safe_import_scipy_stats()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _nan_if_empty(arr):
        return np.asarray(arr).ravel()

    lines = []
    lines.append("label\tkw_H\tkw_p\tmw_train_val_p\tmw_train_test_p\tmw_val_test_p\tmw_p_adj_note\tks_train_val_p\tks_train_test_p\tks_val_test_p")
    for j, name in enumerate(label_names):
        a = _nan_if_empty(y_train[:, j] if y_train.size else np.asarray([]))
        b = _nan_if_empty(y_val[:, j] if y_val.size else np.asarray([]))
        c = _nan_if_empty(y_test[:, j] if y_test.size else np.asarray([]))

        if a.size == 0 or b.size == 0 or c.size == 0:
            lines.append(f"{name}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
            continue

        kw = stats.kruskal(a, b, c)
        # Mann-Whitney U，双侧
        mw_ab = stats.mannwhitneyu(a, b, alternative="two-sided")
        mw_ac = stats.mannwhitneyu(a, c, alternative="two-sided")
        mw_bc = stats.mannwhitneyu(b, c, alternative="two-sided")

        # Bonferroni for 3 pairwise tests
        mw_ab_p_adj = min(float(mw_ab.pvalue) * 3.0, 1.0)
        mw_ac_p_adj = min(float(mw_ac.pvalue) * 3.0, 1.0)
        mw_bc_p_adj = min(float(mw_bc.pvalue) * 3.0, 1.0)

        ks_ab = stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
        ks_ac = stats.ks_2samp(a, c, alternative="two-sided", mode="auto")
        ks_bc = stats.ks_2samp(b, c, alternative="two-sided", mode="auto")

        note = "mw_p为原始p；建议结合Bonferroni校正: p_adj=p*3"
        lines.append(
            f"{name}\t{float(kw.statistic):.6f}\t{float(kw.pvalue):.6g}"
            f"\t{float(mw_ab.pvalue):.6g}\t{float(mw_ac.pvalue):.6g}\t{float(mw_bc.pvalue):.6g}"
            f"\t{note}"
            f"\t{float(ks_ab.pvalue):.6g}\t{float(ks_ac.pvalue):.6g}\t{float(ks_bc.pvalue):.6g}"
        )

        # 另外把Bonferroni校正后的p写到同一文件的附加行，便于直接判读
        lines.append(
            f"{name}__mw_bonferroni_adj\tNA\tNA"
            f"\t{mw_ab_p_adj:.6g}\t{mw_ac_p_adj:.6g}\t{mw_bc_p_adj:.6g}"
            f"\tBonferroni(\u00d73)"
            f"\tNA\tNA\tNA"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"已保存: {out_path}")


def plot_split_distributions(y_train, y_val, y_test, out_path, label_names, bins=40, density=True):
    # 每个label一个子图；叠加train/val/test直方图
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    splits = [
        ("train", y_train, "tab:blue"),
        ("val", y_val, "tab:orange"),
        ("test", y_test, "tab:green"),
    ]

    n_labels = len(label_names)
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5))
    axes = axes.ravel()

    for j in range(n_labels):
        ax = axes[j]
        # 统一bins范围，避免不同split看起来不一致
        all_v = []
        for _, y, _ in splits:
            if y.size:
                all_v.append(y[:, j])
        if not all_v:
            ax.set_title(f"{label_names[j]} (empty)")
            ax.axis("off")
            continue

        all_v = np.concatenate(all_v, axis=0)
        vmin, vmax = float(np.min(all_v)), float(np.max(all_v))
        if vmin == vmax:
            vmax = vmin + 1e-6

        for name, y, color in splits:
            if not y.size:
                continue
            v = y[:, j]
            ax.hist(
                v,
                bins=bins,
                range=(vmin, vmax),
                density=density,
                alpha=0.35,
                color=color,
                label=name,
            )

        ax.set_title(label_names[j])
        ax.set_xlabel("value")
        ax.set_ylabel("density" if density else "count")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"已保存: {out_path}")


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


def save_split_labels_txt(y_train, y_val, y_test, out_path, label_names):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    def _add(split_name, y):
        if y is None or np.size(y) == 0:
            return
        y = np.asarray(y).reshape(-1, len(label_names)).astype(np.float32)
        for i in range(y.shape[0]):
            rows.append([split_name] + [f"{v:.6f}" for v in y[i].tolist()])

    _add("train", y_train)
    _add("val", y_val)
    _add("test", y_test)

    header = ["split"] + label_names
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(r) + "\n")

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

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    processor = Process(model_path=None)
    dataset = ViTacDataset(
        args.data_dir,
        args.label_dir,
        processor=processor,
        augment=args.augment,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    N = len(dataset)
    train_len = int(0.6 * N)
    val_len = int(0.2 * N)
    test_len = N - train_len - val_len
    train_fixed, val_fixed, test_fixed = random_split(
        dataset,
        [train_len, val_len, test_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    # \-\- 评估 train/val/test 分布一致性 \-\-
    label_names = ["SBP", "DBP", "SpO2", "HR"]
    y_train = collect_labels(train_fixed)
    y_val = collect_labels(val_fixed)
    y_test = collect_labels(test_fixed)

    save_split_labels_txt(
        y_train,
        y_val,
        y_test,
        out_path=os.path.join(args.out_dir, "split_y_train_val_test.txt"),
        label_names=label_names,
    )

    plot_split_distributions(
        y_train,
        y_val,
        y_test,
        out_path=os.path.join(args.out_dir, "split_distributions.png"),
        label_names=label_names,
        bins=40,
        density=True,
    )

    stats_train = compute_label_stats(y_train)
    stats_val = compute_label_stats(y_val)
    stats_test = compute_label_stats(y_test)
    save_stats_table(
        out_path=os.path.join(args.out_dir, "split_label_stats.tsv"),
        stats_train=stats_train,
        stats_val=stats_val,
        stats_test=stats_test,
        label_names=label_names,
    )

    distribution_tests(
        y_train,
        y_val,
        y_test,
        out_path=os.path.join(args.out_dir, "split_distribution_tests.tsv"),
        label_names=label_names,
    )

    # \-\- 原有：在 test 上做模型评估 \-\-
    test_loader = DataLoader(test_fixed, batch_size=args.batch_size, shuffle=False)

    # model = Trans(input_len=args.window_size).to(device)
    # model = TransNoDiffAttn(input_len=args.window_size).to(device)
    # model = TransNoDiffAttnSTOnly(input_len=args.window_size).to(device)
    # model = TransNoPressDifAttn(input_len=args.window_size).to(device)

    # model = BiLSTMModel(input_len=args.window_size).to(device)
    model = EncoderOnlyTransformer(input_len=args.window_size).to(device)

    state = torch.load(args.best_path, map_location=device)
    model.load_state_dict(state)

    y_true, y_pred = collect_predictions(model, test_loader, device)

    out_pred_path = os.path.join(args.out_dir, "y_true_y_pred.txt")

    label_names = ["SBP", "DBP", "SpO2", "HR"]
    header_cols = []
    for name in label_names:
        header_cols.append(f"{name}_true")
    for name in label_names:
        header_cols.append(f"{name}_pred")
    for name in label_names:
        header_cols.append(f"{name}_err")

    table = np.concatenate([y_true, y_pred, (y_true - y_pred)], axis=1).astype(np.float32)

    np.savetxt(
        out_pred_path,
        table,
        fmt="%.6f",
        delimiter="\t",
        header="\t".join(header_cols),
        comments="",
    )
    print(f"已保存: {out_pred_path}")

    plot_bland_altman(y_true[:, 0], y_pred[:, 0], os.path.join(args.out_dir, "bland_altman_sbp.png"), title="SBP (mmHg)")
    plot_bland_altman(y_true[:, 1], y_pred[:, 1], os.path.join(args.out_dir, "bland_altman_dbp.png"), title="DBP (mmHg)")
    plot_bland_altman(y_true[:, 2], y_pred[:, 2], os.path.join(args.out_dir, "bland_altman_spo2.png"), title="SpO2 (%)")
    plot_bland_altman(y_true[:, 3], y_pred[:, 3], os.path.join(args.out_dir, "bland_altman_hr.png"), title="HR (bpm)")

    plot_correlation(y_true[:, 0], y_pred[:, 0], os.path.join(args.out_dir, "corr_sbp.png"), title="SBP correlation")
    plot_correlation(y_true[:, 1], y_pred[:, 1], os.path.join(args.out_dir, "corr_dbp.png"), title="DBP correlation")
    plot_correlation(y_true[:, 2], y_pred[:, 2], os.path.join(args.out_dir, "corr_spo2.png"), title="SpO2 correlation")
    plot_correlation(y_true[:, 3], y_pred[:, 3], os.path.join(args.out_dir, "corr_hr.png"), title="HR correlation")

    errors = (y_true - y_pred).astype(np.float32)
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
