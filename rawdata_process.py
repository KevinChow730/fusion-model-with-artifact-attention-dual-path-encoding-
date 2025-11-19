import pandas as pd
import numpy as np
import os


def downsample_to_length_mean(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.ndim != 1:
        raise ValueError("仅支持一维数组。")
    if target_len <= 0:
        raise ValueError("target_len 必须 > 0。")
    if target_len > x.size:
        raise ValueError("target_len 不能大于输入长度（仅支持降采样）。")
    segments = np.array_split(x, target_len)
    return np.array([seg.mean() for seg in segments], dtype=np.float64)


def upsample_to_length_zoh(x: np.ndarray, target_len: int) -> np.ndarray:
    """
    零阶保持上采样到 target_len（或等长）。仅支持上采样/等长，不支持降采样。
    """
    if x.ndim != 1:
        raise ValueError("仅支持一维数组。")
    n = x.size
    if target_len <= 0:
        raise ValueError("target_len 必须 > 0。")
    if target_len < n:
        raise ValueError("仅支持上采样/等长（target_len 不能小于输入长度）。")
    # 采用 floor 映射，保证分段常值；末端用 1e-8 避免越界
    idx = np.floor(np.linspace(0, n - 1e-8, target_len)).astype(np.int64)
    return x[idx]


def _last_valid_before(df: pd.DataFrame, col_idx: int, end_row: int):
    if end_row <= 0:
        return None
    arr = pd.to_numeric(df.iloc[:end_row, col_idx], errors='coerce').to_numpy()
    mask = np.isfinite(arr)
    if mask.any():
        return arr[np.where(mask)[0][-1]]
    return None


def clean_values(arr: np.ndarray, prev_seed: float | None, rel_thresh: float = 0.8) -> np.ndarray:
    """
    规则：
    - NaN 用前一个有效值填充；
    - 若 |x_i - prev| / (|prev| + 1e-12) > rel_thresh（默认 0.8），用 prev 替换；
    - prev 为 0 时，任意非 0 跳变都会被视为超过阈值。
    """
    out = np.full(arr.size, np.nan, dtype=np.float64)
    prev = prev_seed if (prev_seed is not None and np.isfinite(prev_seed)) else None
    eps = 1e-12

    for i in range(arr.size):
        x = arr[i]
        if not np.isfinite(x):
            if prev is not None:
                out[i] = prev
            else:
                # 暂时留空，稍后用首个有效值回填
                continue
        else:
            if prev is None:
                out[i] = x
                prev = x
            else:
                ratio = abs(x - prev) / (abs(prev) + eps)
                if ratio > rel_thresh:
                    out[i] = prev
                else:
                    out[i] = x
                    prev = out[i]

    # 若开头仍有 NaN，用第一个有效值回填
    if not np.isfinite(out[0]):
        finite_idx = np.where(np.isfinite(out))[0]
        if finite_idx.size == 0:
            raise ValueError("清洗失败：序列不存在有效数值。")
        out[:finite_idx[0]] = out[finite_idx[0]]
    return out


def replace_out_of_range_with_prev(arr: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    将 arr 中不在 [low, high] 的点替换为其前一个落在该区间内的数值。
    若在当前位置之前不存在正常值，则保留原值。
    """
    if arr.ndim != 1:
        raise ValueError("仅支持一维数组。")
    out = arr.astype(np.float64, copy=True)
    has_prev = False
    prev = 0.0
    for i in range(out.size):
        v = out[i]
        if np.isfinite(v) and (low <= v <= high):
            prev = v
            has_prev = True
        else:
            if has_prev:
                out[i] = prev
    return out


def load_data(file_path="./rawdata/0826"):
    cnap_path = os.path.join(file_path, "cnap_beats.csv")
    vofa_path = os.path.join(file_path, "vofa.csv")

    if not os.path.exists(cnap_path):
        raise FileNotFoundError(f"文件不存在: {cnap_path}")
    if not os.path.exists(vofa_path):
        raise FileNotFoundError(f"文件不存在: {vofa_path}")

    try:
        # 1) 读取 cnap 并清洗
        cnap_df = pd.read_csv(cnap_path)
        start = 96

        # 转数值（无法解析的置 NaN）
        val_cols = [1, 3, 4]  # SBP, DBP, HR
        cnap_df.iloc[:, val_cols] = cnap_df.iloc[:, val_cols].apply(pd.to_numeric, errors="coerce")

        # 取种子（start 之前最后一个有效值）
        seed_sbp = _last_valid_before(cnap_df, 1, start)
        seed_dbp = _last_valid_before(cnap_df, 3, start)
        seed_hr  = _last_valid_before(cnap_df, 4, start)

        # 提取并清洗：空值与>80%突变用前值填充
        raw_SBP = cnap_df.iloc[start:, 1].to_numpy(dtype=np.float64)
        raw_DBP = cnap_df.iloc[start:, 3].to_numpy(dtype=np.float64)
        raw_HR  = cnap_df.iloc[start:, 4].to_numpy(dtype=np.float64)

        raw_SBP = clean_values(raw_SBP, prev_seed=seed_sbp, rel_thresh=0.8)
        raw_DBP = clean_values(raw_DBP, prev_seed=seed_dbp, rel_thresh=0.8)
        raw_HR  = clean_values(raw_HR,  prev_seed=seed_hr,  rel_thresh=0.8)

        # 2) 读取 vofa 获取 ppg/pressure 及目标长度
        vofa_df = pd.read_csv(vofa_path)
        pressure = vofa_df.iloc[:, 0].to_numpy()
        ppg = vofa_df.iloc[:, 1].to_numpy()
        target_len = len(ppg)

        # 3) 将 SBP/DBP/HR 通过零阶保持上采样到 len(ppg)
        SBP = upsample_to_length_zoh(raw_SBP, target_len)
        DBP = upsample_to_length_zoh(raw_DBP, target_len)
        HR  = upsample_to_length_zoh(raw_HR,  target_len)

        SBP = replace_out_of_range_with_prev(SBP, low=80.0, high=200.0)
        DBP = replace_out_of_range_with_prev(DBP, low=30.0, high=110.0)

        print("数据加载成功:")
        print(f"SBP数组形状: {SBP.shape}, 数据类型: {SBP.dtype}")
        print(f"DBP数组形状: {DBP.shape}, 数据类型: {DBP.dtype}")
        print(f"HR数组形状: {HR.shape}, 数据类型: {HR.dtype}")
        print(f"ppg数组形状: {ppg.shape}, 数据类型: {ppg.dtype}")
        print(f"pressure数组形状: {pressure.shape}, 数据类型: {pressure.dtype}")

        return SBP, DBP, HR, ppg, pressure

    except Exception as e:
        raise ValueError(f"读取文件时出错: {e}")


if __name__ == "__main__":
    SBP, DBP, HR, ppg, pressure = load_data()
    label_out_path = "./dataset/0826/label1.txt"
    data_out_path = "./dataset/0826/data1.txt"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(label_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(data_out_path), exist_ok=True)

    # 按列拼接为 N×3 矩阵并保存，每行：SBP,DBP,HR
    label = np.column_stack((SBP, DBP, HR))
    np.savetxt(label_out_path, label, fmt="%.6f", delimiter=" ", newline="\n")
    print(f"已保存到: {label_out_path}，行数: {label.shape[0]}")


    data = np.column_stack((ppg, pressure))
    np.savetxt(data_out_path, data, fmt="%.6f", delimiter=" ", newline="\n")
    print(f"已保存到: {data_out_path}，行数: {data.shape[0]}")

    print("\n数据统计:")
    print(f"SBP - 范围: [{SBP.min():.4f}, {SBP.max():.4f}], 平均值: {SBP.mean():.4f}, 数据量：{len(SBP)}")
    print(f"DBP - 范围: [{DBP.min():.4f}, {DBP.max():.4f}], 平均值: {DBP.mean():.4f}, 数据量：{len(DBP)}")
    print(f"HR - 范围: [{HR.min():.4f}, {HR.max():.4f}], 平均值: {HR.mean():.4f}, 数据量：{len(HR)}")
    print(f"ppg - 范围: [{ppg.min():.4f}, {ppg.max():.4f}], 平均值: {ppg.mean():.4f}, 数据量：{len(ppg)}")
    print(f"pressure - 范围: [{pressure.min():.4f}, {pressure.max():.4f}], 平均值: {pressure.mean():.4f}, 数据量：{len(pressure)}")