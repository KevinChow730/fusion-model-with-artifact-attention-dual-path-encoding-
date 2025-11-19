import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from process import Process
from sklearn.metrics import r2_score
from scipy import stats


matplotlib.use("Agg")  # 使用无界面后端


def sliding_vad_plot(data_path: str, model_path: str, result_dir: str, label_path: str):
    # 确保结果目录存在
    os.makedirs(result_dir, exist_ok=True)

    # 加载模型
    pred_process = Process(model_path=model_path)

    # 读取数据文件（假设是两列的txt文件）
    try:
        data = np.loadtxt(data_path)  # 加载数据文件
        if data.ndim == 1:
            raise ValueError("数据文件应该包含两列")
        if data.shape[1] != 2:
            raise ValueError(f"期望2列数据，得到{data.shape[1]}列")
    except Exception as e:
        raise ValueError(f"Error loading data file: {e}")

    # 滑动窗口参数
    window_size = 128  # 窗口大小
    step_size = 64  # 步长
    num_windows = (len(data) - window_size) // step_size + 1

    pred_results = []  # 存储所有预测结果

    print(f"数据总行数: {len(data)}")
    print(f"滑动窗口数量: {num_windows}")
    print("开始预测...")

    # 滑窗处理
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size

        # 提取窗口数据 [128, 2]
        window_data = data[start:end]

        # 转置为 [2, 128] 格式（与训练时一致）
        window_data = window_data.T

        # 使用模型进行预测
        pred = pred_process.process_window(window_data)  # 返回 [max_val, min_val]
        pred_results.append(pred)

        if (i + 1) % 100 == 0:
            print(f"已处理: {i + 1}/{num_windows} 窗口")

    # 转换为numpy数组
    pred_results = np.array(pred_results)  # [num_windows, 2]

    print(f"预测完成！共生成 {len(pred_results)} 个预测结果")
    print(f"每个预测包含2个值：[最大值, 最小值]")

    # 构造时间轴（以窗口为单位）
    window_time = np.arange(len(pred_results))

    # 真实标签
    true_labels = None
    try:
        true_labels = np.loadtxt(label_path)  # 加载真实标签文件
        max_col1 = np.max(true_labels[:, 0])  # 第一列最大值
        min_col1 = np.min(true_labels[:, 0])  # 第一列最小值
        max_col2 = np.max(true_labels[:, 1])  # 第二列最大值
        min_col2 = np.min(true_labels[:, 1])  # 第二列最小值

        print(f"第一列范围: 最小值={min_col1}, 最大值={max_col1}")
        print(f"第二列范围: 最小值={min_col2}, 最大值={max_col2}")
    except Exception as e:
        print(f"无法加载真实标签文件 {label_path}: {e}")

    # 确保预测结果和真实标签长度匹配
    if true_labels is not None:
        min_len = min(len(pred_results), len(true_labels))
        pred_results = pred_results[:min_len]
        true_labels = true_labels[:min_len]
        window_time = window_time[:min_len]
        print(f"对齐后的数据长度: {min_len}")

    # 图1：原始数据 + 预测/真实 + 误差 + Bland-Altman
    plt.figure(figsize=(10, 8))
    # 子图1：原始数据的两列
    plt.subplot(3, 2, 1)
    data_time = np.arange(len(data))
    plt.plot(data_time, data[:, 0], label="PPG", alpha=0.7)
    plt.plot(data_time, data[:, 1], label="Motion", alpha=0.7)
    plt.xlabel("Data Point")
    plt.ylabel("Value")
    plt.title("Original Data")
    plt.legend()
    plt.grid(True)

    # 子图2：预测的最大值 vs 真实最大值
    plt.subplot(3, 2, 2)
    plt.plot(window_time, pred_results[:, 0], label="Predicted SBP", color='red', linewidth=2)
    if true_labels is not None:
        plt.plot(window_time, true_labels[:, 0], label="True SBP", color='darkred', linewidth=2, linestyle='--')
    plt.xlabel("Window Index")
    plt.ylabel("Max Value")
    plt.title("Maximum Values: Predicted vs True")
    plt.legend()
    plt.grid(True)

    # 子图3：预测的最小值 vs 真实最小值
    plt.subplot(3, 2, 3)
    plt.plot(window_time, pred_results[:, 1], label="Predicted DBP", color='blue', linewidth=2)
    if true_labels is not None:
        plt.plot(window_time, true_labels[:, 1], label="True DBP", color='darkblue', linewidth=2, linestyle='--')
    plt.xlabel("Window Index")
    plt.ylabel("Min Value")
    plt.title("Minimum Values: Predicted vs True")
    plt.legend()
    plt.grid(True)

    # 子图4：误差分析
    if true_labels is not None:
        plt.subplot(3, 2, 4)
        error_max = pred_results[:, 0] - true_labels[:, 0]
        error_min = pred_results[:, 1] - true_labels[:, 1]
        plt.plot(window_time, error_max, label="Max Error", color='red', alpha=0.7)
        plt.plot(window_time, error_min, label="Min Error", color='blue', alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel("Window Index")
        plt.ylabel("Error (Predicted - True)")
        plt.title("Prediction Errors")
        plt.legend()
        plt.grid(True)

        # Bland-Altman图 - SBP
        plt.subplot(3, 2, 5)
        mean_sbp = (pred_results[:-5, 0] + true_labels[:-5, 0]) / 2
        diff_sbp = pred_results[:-5, 0] - true_labels[:-5, 0]
        mean_diff_sbp = np.mean(diff_sbp)
        std_diff_sbp = np.std(diff_sbp)
        plt.scatter(mean_sbp, diff_sbp, alpha=0.6, color='red')
        plt.axhline(y=mean_diff_sbp, color='red', linestyle='-', label=f'Mean Diff: {mean_diff_sbp:.3f}')
        plt.axhline(y=mean_diff_sbp + 1.96 * std_diff_sbp, color='red', linestyle='--',
                    label=f'+1.96SD: {mean_diff_sbp + 1.96 * std_diff_sbp:.3f}')
        plt.axhline(y=mean_diff_sbp - 1.96 * std_diff_sbp, color='red', linestyle='--',
                    label=f'-1.96SD: {mean_diff_sbp - 1.96 * std_diff_sbp:.3f}')
        plt.xlabel('Mean of Predicted and True SBP')
        plt.ylabel('Difference (Predicted - True)')
        plt.title('Bland-Altman Plot - SBP')
        plt.legend()
        plt.grid(True)

        # Bland-Altman图 - DBP
        plt.subplot(3, 2, 6)
        mean_dbp = (pred_results[:-5, 1] + true_labels[:-5, 1]) / 2
        diff_dbp = pred_results[:-5, 1] - true_labels[:-5, 1]
        mean_diff_dbp = np.mean(diff_dbp)
        std_diff_dbp = np.std(diff_dbp)
        plt.scatter(mean_dbp, diff_dbp, alpha=0.6, color='blue')
        plt.axhline(y=mean_diff_dbp, color='blue', linestyle='-', label=f'Mean Diff: {mean_diff_dbp:.3f}')
        plt.axhline(y=mean_diff_dbp + 1.96 * std_diff_dbp, color='blue', linestyle='--',
                    label=f'+1.96SD: {mean_diff_dbp + 1.96 * std_diff_dbp:.3f}')
        plt.axhline(y=mean_diff_dbp - 1.96 * std_diff_dbp, color='blue', linestyle='--',
                    label=f'-1.96SD: {mean_diff_dbp - 1.96 * std_diff_dbp:.3f}')
        plt.xlabel('Mean of Predicted and True DBP')
        plt.ylabel('Difference (Predicted - True)')
        plt.title('Bland-Altman Plot - DBP')
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    fig1_path = os.path.normpath(os.path.join(result_dir, "overview_analysis.png"))
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"图像已保存: {fig1_path}")

    # 图2：相关性分析（仅当有真实标签时）
    if true_labels is not None:
        plt.figure(figsize=(12, 5))

        # SBP散点图和拟合直线
        plt.subplot(1, 2, 1)
        pred_sbp = pred_results[:, 0]
        true_sbp = true_labels[:, 0]
        correlation_sbp, p_value_sbp = stats.pearsonr(pred_sbp, true_sbp)
        r2_sbp = r2_score(true_sbp, pred_sbp)
        slope_sbp, intercept_sbp, _, _, _ = stats.linregress(true_sbp, pred_sbp)
        plt.scatter(true_sbp, pred_sbp, alpha=0.6, color='red', s=20)
        x_line = np.array([np.min(true_sbp), np.max(true_sbp)])
        y_line = slope_sbp * x_line + intercept_sbp
        plt.plot(x_line, y_line, 'red', linewidth=2, label=f'y={slope_sbp:.3f}x+{intercept_sbp:.3f}')
        plt.plot(x_line, x_line, 'black', linestyle='--', alpha=0.5, label='y=x')
        plt.xlabel('True SBP')
        plt.ylabel('Predicted SBP')
        plt.title(f'SBP Correlation Analysis\nR²={r2_sbp:.4f}, r={correlation_sbp:.4f}, p={p_value_sbp:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # DBP散点图和拟合直线
        plt.subplot(1, 2, 2)
        pred_dbp = pred_results[:, 1]
        true_dbp = true_labels[:, 1]
        correlation_dbp, p_value_dbp = stats.pearsonr(pred_dbp, true_dbp)
        r2_dbp = r2_score(true_dbp, pred_dbp)
        slope_dbp, intercept_dbp, _, _, _ = stats.linregress(true_dbp, pred_dbp)
        plt.scatter(true_dbp, pred_dbp, alpha=0.6, color='blue', s=20)
        x_line = np.array([np.min(true_dbp), np.max(true_dbp)])
        y_line = slope_dbp * x_line + intercept_dbp
        plt.plot(x_line, y_line, 'blue', linewidth=2, label=f'y={slope_dbp:.3f}x+{intercept_dbp:.3f}')
        plt.plot(x_line, x_line, 'black', linestyle='--', alpha=0.5, label='y=x')
        plt.xlabel('True DBP')
        plt.ylabel('Predicted DBP')
        plt.title(f'DBP Correlation Analysis\nR²={r2_dbp:.4f}, r={correlation_dbp:.4f}, p={p_value_dbp:.4f}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        fig2_path = os.path.normpath(os.path.join(result_dir, "correlation_analysis.png"))
        plt.savefig(fig2_path, dpi=300)
        plt.close()
        print(f"图像已保存: {fig2_path}")

        # 打印相关性统计信息
        print(f"\n相关性分析结果:")
        print(f"SBP:")
        print(f"  皮尔逊相关系数: {correlation_sbp:.6f}")
        print(f"  R²决定系数: {r2_sbp:.6f}")
        print(f"  p值: {p_value_sbp:.6f}")
        print(f"  拟合方程: y = {slope_sbp:.6f}x + {intercept_sbp:.6f}")

        print(f"DBP:")
        print(f"  皮尔逊相关系数: {correlation_dbp:.6f}")
        print(f"  R²决定系数: {r2_dbp:.6f}")
        print(f"  p值: {p_value_dbp:.6f}")
        print(f"  拟合方程: y = {slope_dbp:.6f}x + {intercept_dbp:.6f}")

    # 添加Bland-Altman统计信息
    if true_labels is not None:
        print(f"\nBland-Altman分析:")
        print(f"SBP:")
        print(f"  平均差值: {mean_diff_sbp:.6f}")
        print(f"  差值标准差: {std_diff_sbp:.6f}")
        print(f"  95%一致性界限: [{mean_diff_sbp - 1.96 * std_diff_sbp:.6f}, {mean_diff_sbp + 1.96 * std_diff_sbp:.6f}]")

        print(f"DBP:")
        print(f"  平均差值: {mean_diff_dbp:.6f}")
        print(f"  差值标准差: {std_diff_dbp:.6f}")
        print(f"  95%一致性界限: [{mean_diff_dbp - 1.96 * std_diff_dbp:.6f}, {mean_diff_dbp + 1.96 * std_diff_dbp:.6f}]")

    # 写入预测结果到文件
    output_path = os.path.normpath(os.path.join(result_dir, 'prediction_results.txt'))
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Window_Index Max_Value Min_Value\n")
            for i, (max_val, min_val) in enumerate(pred_results):
                f.write(f"{i} {max_val:.6f} {min_val:.6f}\n")
        print(f"预测结果已保存到: {output_path}")
    except Exception as e:
        raise ValueError(f"Error writing to output file: {e}")


if __name__ == "__main__":
    data_path = "./testdata/ppg_pressure_4.txt"  # 改为你的数据文件
    model_path = "./model/bp_artifact.pth"       # model_5: SBP:0±5, dbp:3±5
    label_path = "./testlabel/envelope_4.txt"
    result_dir = "./"
    sliding_vad_plot(data_path, model_path, result_dir, label_path)
