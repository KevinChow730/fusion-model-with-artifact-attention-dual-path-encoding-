import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")  # 或 'Qt5Agg'
from itertools import cycle


# 绘制函数
def plot_data(data, title, subplot_index):
    mae = [item["mae"] for item in data]
    mse = [item["mse"] for item in data]
    labels = [item["label"] for item in data]

    # 定义颜色循环
    colors = cycle(['blue', 'green', 'orange', 'purple', 'brown'])

    plt.subplot(1, 2, subplot_index)
    for i, label in enumerate(labels):
        if label == "TacBP":
            plt.scatter(mae[i], mse[i], color='red', marker='*', s=150, label=label if i == 0 else "")
        else:
            plt.scatter(mae[i], mse[i], color=next(colors), alpha=0.7, label=label if i == 0 else "")
        plt.text(mae[i], mse[i], label, fontsize=10, ha='right', va='bottom')

    plt.xlabel("MAE")
    plt.ylabel("MSE")
    plt.title(title)
    plt.grid(True)


if __name__ == "__main__":
    # 示例数据
    SBP = [
        {"label": "TacBP", "mae": 0.7608, "mse": 1.0616},
        {"label": "CNN", "mae": 5.3706, "mse": 45.5107},
        {"label": "RNN", "mae": 8.7421, "mse": 110.5697},
        {"label": "LSTM", "mae": 7.2079, "mse": 64.1960},
        {"label": "Transformer", "mae": 9.9974, "mse": 136.2062},
    ]

    DBP = [
        {"label": "TacBP", "mae": 0.6583, "mse": 0.7501},
        {"label": "CNN", "mae": 4.0915, "mse": 25.3723},
        {"label": "RNN", "mae": 3.4107, "mse": 15.4630},
        {"label": "LSTM", "mae": 3.1205, "mse": 13.9138},
        {"label": "Transformer", "mae": 3.2878, "mse": 17.2820},
    ]
    # 创建图形
    plt.figure(figsize=(6, 3))

    # 绘制SBP数据
    plot_data(SBP, "SBP Scatter Plot", 1)

    # 绘制DBP数据
    plot_data(DBP, "DBP Scatter Plot", 2)

    # 显示图形
    plt.tight_layout()
    plt.show()