import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from diff_attn import CrossDiffAttention


def make_signals(T: int = 200, noise_start: int = 80, noise_end: int = 120, noise_std: float = 1.0):
    t = np.linspace(0.0, 2.0 * np.pi, T, endpoint=False)
    clean = np.sin(3.0 * t).astype(np.float32)

    k = clean.copy()
    noise = np.zeros_like(k)
    noise[noise_start:noise_end] = np.random.normal(0.0, noise_std, size=(noise_end - noise_start)).astype(np.float32)
    k_noisy = k + noise

    q = clean + noise
    return clean, q, k_noisy, (noise_start, noise_end)


def plot_heatmap(ax, mat, title: str, vmin=None, vmax=None):
    im = ax.imshow(mat, aspect="auto", origin="lower", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Key index")
    ax.set_ylabel("Query index")
    return im


@torch.no_grad()
def extract_attn(attn: CrossDiffAttention, q_input: torch.Tensor, k_input: torch.Tensor):
    """
    与 `CrossDiffAttention.forward` 同步的注意力提取版本（不改模块源码）。
    返回:
      attn1/attn2/diff_attn: [B, Eh, T, T]
      v: [B, Eh, T, 2Dh]
      lambda_full: 标量张量
    """
    B, N, _ = q_input.shape

    q = attn.q_proj(q_input)
    k = attn.k_proj(k_input)
    v_raw = attn.v_proj(k_input)

    Eh = attn.effective_heads
    Dh = attn.head_dim

    q = q.view(B, N, 2 * Eh, Dh).transpose(1, 2)          # [B, 2Eh, T, Dh]
    k = k.view(B, N, 2 * Eh, Dh).transpose(1, 2)          # [B, 2Eh, T, Dh]
    v = v_raw.view(B, N, Eh, 2 * Dh).transpose(1, 2)      # [B, Eh, T, 2Dh]

    q = q * attn.scaling
    attn_scores = torch.matmul(q, k.transpose(-1, -2))    # [B, 2Eh, T, T]
    attn_probs = torch.softmax(attn_scores, dim=-1)
    attn_probs = attn.attn_drop(attn_probs)

    attn_probs = attn_probs.view(B, Eh, 2, N, N)          # [B, Eh, 2, T, T]

    lambda_1 = torch.exp(torch.sum(attn.lambda_q1 * attn.lambda_k1))
    lambda_2 = torch.exp(torch.sum(attn.lambda_q2 * attn.lambda_k2))
    lambda_full = lambda_1 - lambda_2 + attn.lambda_init

    attn1 = attn_probs[:, :, 0, :, :]                     # [B, Eh, T, T]
    attn2 = attn_probs[:, :, 1, :, :]                     # [B, Eh, T, T]
    diff_attn = attn1 - lambda_full * attn2               # [B, Eh, T, T]

    return attn1, attn2, diff_attn, v, lambda_full


@torch.no_grad()
def reconstruct_1d_from_attn(diff_attn: torch.Tensor, v: torch.Tensor):
    """
    用 diff_attn 对 V 做加权求和 -> 得到每个时间步的特征 -> readout 成 1D 序列。
    输入:
      diff_attn: [B, Eh, T, T]
      v: [B, Eh, T, 2Dh]
    输出:
      recon_1d: [B, T]（简单均值 readout）
    """
    attn_out = torch.matmul(diff_attn, v)                 # [B, Eh, T, 2Dh]
    recon_1d = attn_out.mean(dim=-1).mean(dim=1)          # 先对特征维平均，再对 head 平均 -> [B, T]
    return recon_1d


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    T = 200
    d_model = 256
    num_heads = 16

    clean_1d, q_1d, k_1d, (ns, ne) = make_signals(T=T, noise_start=80, noise_end=120, noise_std=1.2)

    proj = torch.randn(1, 1, d_model) / np.sqrt(d_model)
    q = torch.from_numpy(q_1d)[None, :, None].float() * proj  # [1, T, d_model]
    k = torch.from_numpy(k_1d)[None, :, None].float() * proj  # [1, T, d_model]

    attn = CrossDiffAttention(dim=d_model, num_heads=num_heads, attn_drop=0.0, proj_drop=0.0, lambda_init=0.8)
    attn.eval()

    attn1_t, attn2_t, diff_t, v_t, lam_t = extract_attn(attn, q, k)
    recon_t = reconstruct_1d_from_attn(diff_t, v_t)  # [1, T]

    attn1 = attn1_t[0].mean(0).cpu().numpy()  # [T, T]
    attn2 = attn2_t[0].mean(0).cpu().numpy()  # [T, T]
    diff = diff_t[0].mean(0).cpu().numpy()    # [T, T]
    lam = float(lam_t.cpu().item())

    recon_1d = recon_t[0].cpu().numpy()

    # 统一热力图色条范围（对称，兼容 diff 的负值）
    max_abs = float(np.max(np.abs(np.stack([attn1, attn2, diff], axis=0))))
    vmin, vmax = -max_abs, max_abs

    # 为了对比更直观：把 recon 做一个线性归一到 clean 的幅值范围（只用于画图，不改变机制）
    r_mean, r_std = float(recon_1d.mean()), float(recon_1d.std() + 1e-8)
    c_mean, c_std = float(clean_1d.mean()), float(clean_1d.std() + 1e-8)
    recon_vis = (recon_1d - r_mean) / r_std * c_std + c_mean

    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    im0 = plot_heatmap(ax0, attn1, "attn1\\ \\(mean\\ over\\ heads\\)", vmin=vmin, vmax=vmax)
    im1 = plot_heatmap(ax1, attn2, "attn2\\ \\(mean\\ over\\ heads\\)", vmin=vmin, vmax=vmax)
    im2 = plot_heatmap(
        ax2,
        diff,
        f"diff\\_attn\\ =\\ attn1\\ -\\ lambda\\_full\\*attn2\\n\\(lambda\\_full={lam:.3f}\\)",
        vmin=vmin,
        vmax=vmax,
    )

    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 输入信号
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(clean_1d, label="clean\\ \\(sin\\)", linewidth=1.2)
    ax3.plot(q_1d, label="Q\\ =\\ clean\\ +\\ noise", linewidth=1.2, alpha=0.9)
    ax3.plot(k_1d, label="K\\ =\\ clean\\ +\\ noise\\(segment\\)", linewidth=1.2, alpha=0.85)
    ax3.axvspan(ns, ne, color="red", alpha=0.10, label="noise\\ segment")
    ax3.set_title("Input\\ signals")
    ax3.set_xlabel("time\\ step")
    ax3.legend(loc="upper right")

    # 还原与对比
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(clean_1d, label="clean\\ \\(target\\)", linewidth=1.5)
    ax4.plot(k_1d, label="K\\ noisy\\ \\(source\\)", linewidth=1.0, alpha=0.8)
    ax4.plot(recon_vis, label="recon\\ from\\ diff\\_attn\\ \\(scaled\\ for\\ display\\)", linewidth=1.5)
    ax4.axvspan(ns, ne, color="red", alpha=0.10)
    ax4.set_title("Reconstruction\\ from\\ differential\\ attention")
    ax4.set_xlabel("time\\ step")
    ax4.legend(loc="upper right")

    plt.tight_layout()

    out_path = Path.cwd() / "diff_attn_viz.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved\\ figure\\ to:\\ {out_path}")


if __name__ == "__main__":
    main()
