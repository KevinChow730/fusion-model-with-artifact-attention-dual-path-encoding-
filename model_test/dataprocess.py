import pandas as pd
import numpy as np
import os
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

def load_data(file_path="./rawdata/0826"):
    """
    从rawdata文件夹读取cnap.csv和vofa.csv文件

    Returns:
        bp: cnap.csv第二列的numpy数组
        ppg: vofa.csv第一列的numpy数组
        pressure: vofa.csv第二列的numpy数组
    """

    # 构建文件路径
    cnap_path = os.path.join(file_path, "cnap.csv")  # 100 Hz -> 降采样至25hz，每个点代表4s内平均值
    vofa_path = os.path.join(file_path, "vofa.csv")  # 33 Hz -> 128个点, 4s预测一次

    # 检查文件是否存在
    if not os.path.exists(cnap_path):
        raise FileNotFoundError(f"文件不存在: {cnap_path}")
    if not os.path.exists(vofa_path):
        raise FileNotFoundError(f"文件不存在: {vofa_path}")

    try:
        # 读取cnap.csv并提取第二列
        cnap_df = pd.read_csv(cnap_path)
        if cnap_df.shape[1] < 2:
            raise ValueError(f"cnap.csv列数不足，需要至少2列，实际有{cnap_df.shape[1]}列")
        bp = cnap_df.iloc[:, 1].values  # 第二列（索引1）

        # 读取vofa.csv并提取第一列和第二列
        vofa_df = pd.read_csv(vofa_path)
        if vofa_df.shape[1] < 2:
            raise ValueError(f"vofa.csv列数不足，需要至少2列，实际有{vofa_df.shape[1]}列")
        pressure = vofa_df.iloc[:, 0].values      # 第一列（索引0）
        ppg = vofa_df.iloc[:, 1].values # 第二列（索引1）

        print(f"数据加载成功:")
        print(f"bp数组形状: {bp.shape}, 数据类型: {bp.dtype}")
        print(f"ppg数组形状: {ppg.shape}, 数据类型: {ppg.dtype}")
        print(f"pressure数组形状: {pressure.shape}, 数据类型: {pressure.dtype}")

        return bp, ppg, pressure

    except Exception as e:
        raise ValueError(f"读取文件时出错: {e}")


def envelope_extraction(signal, dbp=40, sbp=150, sensitivity='very_high', sample=None):
    """
    针对三角波信号优化的包络提取，过滤异常点

    Args:
        signal: 输入信号
        dbp: 舒张压下限（默认40）
        sbp: 收缩压上限（默认150）
        sensitivity: 敏感度 ('low', 'medium', 'high', 'very_high')
        sample: 重采样点数，如果为None则不重采样
    """

    # 根据敏感度设置参数
    sensitivity_params = {
        'low': {
            'distance_factor': 100,
            'height_factor': 0.5,
            'prominence_factor': 0.3
        },
        'medium': {
            'distance_factor': 50,
            'height_factor': 0.3,
            'prominence_factor': 0.2
        },
        'high': {
            'distance_factor': 30,
            'height_factor': 0.2,
            'prominence_factor': 0.15
        },
        'very_high': {
            'distance_factor': 20,
            'height_factor': 0.1,
            'prominence_factor': 0.1
        }
    }

    params = sensitivity_params[sensitivity]

    # 自适应参数设置
    signal_length = len(signal)
    min_distance = max(10, signal_length // params['distance_factor'])

    # 动态调整height阈值
    signal_std = np.std(signal)
    signal_mean = np.mean(signal)

    # 找到所有峰值和谷值
    peaks_max, properties_max = find_peaks(signal,
                                         distance=min_distance,
                                         height=signal_mean + params['height_factor'] * signal_std,
                                         prominence=params['prominence_factor'] * signal_std)

    peaks_min, properties_min = find_peaks(-signal,
                                         distance=min_distance,
                                         height=-signal_mean + params['height_factor'] * signal_std,
                                         prominence=params['prominence_factor'] * signal_std)

    # 过滤异常点：保留 dbp <= value <= sbp 的点
    valid_peaks_max = []
    for peak in peaks_max:
        if dbp <= signal[peak] <= sbp:
            valid_peaks_max.append(peak)
    peaks_max = np.array(valid_peaks_max)

    valid_peaks_min = []
    for peak in peaks_min:
        if dbp <= signal[peak] <= sbp:
            valid_peaks_min.append(peak)
    peaks_min = np.array(valid_peaks_min)

    print(f"收缩压上限: {sbp}, 舒张压下限: {dbp}")
    print(f"原始检测到的峰值数量: {len(properties_max['peak_heights']) if 'peak_heights' in properties_max else len(peaks_max)}")
    print(f"过滤后的峰值数量: {len(peaks_max)}")
    print(f"原始检测到的谷值数量: {len(properties_min['peak_heights']) if 'peak_heights' in properties_min else len(peaks_min)}")
    print(f"过滤后的谷值数量: {len(peaks_min)}")

    x = np.arange(len(signal))

    # 上包络插值
    if len(peaks_max) > 1:
        # 添加边界点，但也要检查边界点是否在有效范围内
        boundary_points = []
        if dbp <= signal[0] <= sbp:
            boundary_points.append(0)
        if dbp <= signal[-1] <= sbp:
            boundary_points.append(len(signal)-1)

        # 合并边界点和有效峰值点
        all_peaks = np.unique(np.concatenate((boundary_points, peaks_max)))

        if len(all_peaks) > 1:
            peaks_max_values = signal[all_peaks]
            f_upper = interp1d(all_peaks, peaks_max_values, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
            upper_envelope = f_upper(x)
        else:
            # 如果有效点太少，使用信号中有效范围的最大值
            valid_signal = signal[(signal >= dbp) & (signal <= sbp)]
            if len(valid_signal) > 0:
                upper_envelope = np.full_like(signal, valid_signal.max())
            else:
                upper_envelope = np.full_like(signal, sbp)  # 默认上限
    else:
        valid_signal = signal[(signal >= dbp) & (signal <= sbp)]
        if len(valid_signal) > 0:
            upper_envelope = np.full_like(signal, valid_signal.max())
        else:
            upper_envelope = np.full_like(signal, sbp)

    # 下包络插值
    if len(peaks_min) > 1:
        # 添加边界点，但也要检查边界点是否在有效范围内
        boundary_points = []
        if dbp <= signal[0] <= sbp:
            boundary_points.append(0)
        if dbp <= signal[-1] <= sbp:
            boundary_points.append(len(signal)-1)

        # 合并边界点和有效谷值点
        all_valleys = np.unique(np.concatenate((boundary_points, peaks_min)))

        if len(all_valleys) > 1:
            peaks_min_values = signal[all_valleys]
            f_lower = interp1d(all_valleys, peaks_min_values, kind='linear',
                             bounds_error=False, fill_value='extrapolate')
            lower_envelope = f_lower(x)
        else:
            # 如果有效点太少，使用信号中有效范围的最小值
            valid_signal = signal[(signal >= dbp) & (signal <= sbp)]
            if len(valid_signal) > 0:
                lower_envelope = np.full_like(signal, valid_signal.min())
            else:
                lower_envelope = np.full_like(signal, dbp)  # 默认下限
    else:
        valid_signal = signal[(signal >= dbp) & (signal <= sbp)]
        if len(valid_signal) > 0:
            lower_envelope = np.full_like(signal, valid_signal.min())
        else:
            lower_envelope = np.full_like(signal, dbp)

    # 重采样包络线到指定点数
    if sample is not None and sample != len(signal):
        # 创建新的采样点
        x_original = np.arange(len(signal))
        x_resampled = np.linspace(0, len(signal)-1, sample)

        # 对上包络重采样
        f_upper_resample = interp1d(x_original, upper_envelope, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        upper_envelope = f_upper_resample(x_resampled)

        # 对下包络重采样
        f_lower_resample = interp1d(x_original, lower_envelope, kind='linear',
                                   bounds_error=False, fill_value='extrapolate')
        lower_envelope = f_lower_resample(x_resampled)

        print(f"包络线已重采样: {len(signal)} -> {sample} 个点")

    return upper_envelope, lower_envelope, peaks_max, peaks_min


def analyze_envelope(bp, upper_envelope, lower_envelope):
    """
    分析包络特征
    """
    envelope_width = upper_envelope - lower_envelope

    print(f"\n包络分析:")
    print(f"上包络范围: [{upper_envelope.min():.4f}, {upper_envelope.max():.4f}]")
    print(f"下包络范围: [{lower_envelope.min():.4f}, {lower_envelope.max():.4f}]")
    print(f"包络宽度平均值: {envelope_width.mean():.4f}")
    print(f"包络宽度标准差: {envelope_width.std():.4f}")
    print(f"最大包络宽度: {envelope_width.max():.4f}")
    print(f"最小包络宽度: {envelope_width.min():.4f}")


if __name__ == "__main__":
    try:
        bp, ppg, pressure = load_data()
        resample_rate = int(len(ppg) / 128)

        # 打印数据统计信息
        print(f"\n数据统计:")
        print(f"bp - 范围: [{bp.min():.4f}, {bp.max():.4f}], 平均值: {bp.mean():.4f}")
        print(f"ppg - 范围: [{ppg.min():.4f}, {ppg.max():.4f}], 平均值: {ppg.mean():.4f}, 数据量：{len(ppg)}")
        print(f"pressure - 范围: [{pressure.min():.4f}, {pressure.max():.4f}], 平均值: {pressure.mean():.4f}, 数据量：{len(pressure)}")

        upper_env, lower_env, peaks_max, peaks_min = envelope_extraction(
            bp, dbp=40, sbp=150, sensitivity='low', sample=resample_rate)

        # 保存数据到txt文件
        output_dir = "./"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 定义ppg_pressure的分割索引
        ppg_splits = [(0, 20000), (20000, 38570), (38570, 58570), (58570, len(ppg))]
        # 定义envelope的分割索引
        env_splits = [(0, 156), (156, 301), (301, 457), (457, len(upper_env))]

        # 分割保存ppg和pressure数据
        for i, (start, end) in enumerate(ppg_splits, 1):
            ppg_segment = ppg[start:end]
            pressure_segment = pressure[start:end]
            ppg_pressure_data = np.column_stack((ppg_segment, pressure_segment))

            filename = f"ppg_pressure_{i}.txt"
            np.savetxt(os.path.join(output_dir, filename), ppg_pressure_data, fmt='%.6f')
            print(
                f"ppg和pressure数据段{i}已保存到 {os.path.join(output_dir, filename)} (索引{start}:{end}, {len(ppg_segment)}行)")

        # 分割保存upper_env和lower_env数据
        for i, (start, end) in enumerate(env_splits, 1):
            upper_segment = upper_env[start:end]
            lower_segment = lower_env[start:end]
            envelope_data = np.column_stack((upper_segment, lower_segment))

            filename = f"envelope_{i}.txt"
            np.savetxt(os.path.join(output_dir, filename), envelope_data, fmt='%.6f')
            print(
                f"上包络和下包络数据段{i}已保存到 {os.path.join(output_dir, filename)} (索引{start}:{end}, {len(upper_segment)}行)")

        time_bp = np.arange(len(bp))
        time_env = np.arange(resample_rate)
        time_ppg = np.arange(len(ppg))
        time_pressure = np.arange(len(pressure))

        plt.figure(figsize=(15, 12))  # 增加高度以容纳更多子图

        # 子图1：血压信号和包络
        plt.subplot(3, 1, 1)
        plt.plot(time_bp, bp, 'r-', linewidth=1, alpha=0.8, label='BP Signal')
        plt.plot(time_env / resample_rate * len(bp), upper_env, 'b-', linewidth=2, label='Upper Envelope')
        plt.plot(time_env / resample_rate * len(bp), lower_env, 'g-', linewidth=2, label='Lower Envelope')
        plt.plot(peaks_max, bp[peaks_max], 'bo', markersize=4, label=f'Detected Peaks ({len(peaks_max)})')
        plt.plot(peaks_min, bp[peaks_min], 'go', markersize=4, label=f'Detected Valleys ({len(peaks_min)})')
        plt.title(f'BP with Adaptive Envelope Detection (Sensitivity: {"low"})')
        plt.xlabel('Sample Index')
        plt.ylabel('BP Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图2：PPG信号
        plt.subplot(3, 1, 2)
        plt.plot(time_ppg, ppg, 'purple', linewidth=1, alpha=0.8, label='PPG Signal')
        plt.title('PPG Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('PPG Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图3：压力信号
        plt.subplot(3, 1, 3)
        plt.plot(time_pressure, pressure, 'orange', linewidth=1, alpha=0.8, label='Pressure Signal')
        plt.title('Pressure Signal')
        plt.xlabel('Sample Index')
        plt.ylabel('Pressure Value')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 分析包络特征
        analyze_envelope(bp, upper_env, lower_env)

    except Exception as e:
        print(f"错误: {e}")