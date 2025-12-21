import numpy as np


def generate_sample(seq_len: int = 200):
    """生成一个样本: x1,x2,x3 (各为长度 seq_len 的向量) 以及对应的 y1..y4。"""
    # 随机生成 x1, x2, x3
    x1 = np.random.randn(seq_len)
    x2 = np.random.randn(seq_len)
    x3 = np.random.randn(seq_len)

    # 构造复杂多项式关系
    y1 = 1.2 * x1 ** 2 - 0.7 * x2 * x3 + 0.5 * x1 ** 3 + 0.1 * x2
    y2 = -0.8 * x2 ** 2 + 0.3 * x1 * x3 + 0.2 * x3 ** 3 - 0.4 * x1
    y3 = 0.6 * x3 ** 2 + 0.9 * x1 * x2 - 0.3 * x2 ** 3 + 0.05
    y4 = 0.4 * x1 ** 2 + 0.2 * x2 ** 2 + 0.1 * x3 ** 2 - 0.5 * x1 * x2 + 0.7 * x2 * x3

    return x1, x2, x3, y1, y2, y3, y4


def main(
    num_samples: int = 50,
    seq_len: int = 200,
    data_path: str = "data.txt",
    label_path: str = "label.txt",
):
    """
    生成:
    - data.txt: 共 num_samples 个样本, 每个样本 seq_len 行, 每行 3 个数 (x1,x2,x3)
      => 总行数 = num_samples * seq_len
    - label.txt: 共 num_samples 行, 每行 4 个数 (y1,y2,y3,y4 在该样本上的聚合, 这里用均值)
    """
    with open(data_path, "w", encoding="utf-8") as f_data, \
         open(label_path, "w", encoding="utf-8") as f_label:

        for _ in range(num_samples):
            x1, x2, x3, y1, y2, y3, y4 = generate_sample(seq_len)

            # 写 data.txt: 每个样本占 seq_len 行, 每行 "x1 x2 x3"
            for i in range(seq_len):
                f_data.write(f"{x1[i]:.6f} {x2[i]:.6f} {x3[i]:.6f}\n")

            # 写 label.txt: 对该样本的 y1..y4 做聚合 (这里用均值)
            y1_mean = float(np.mean(y1))
            y2_mean = float(np.mean(y2))
            y3_mean = float(np.mean(y3))
            y4_mean = float(np.mean(y4))
            f_label.write(
                f"{y1_mean:.6f} {y2_mean:.6f} {y3_mean:.6f} {y4_mean:.6f}\n"
            )


if __name__ == "__main__":
    # 生成 50 个样本, 每个样本长度为 200
    main(num_samples=50, seq_len=200,
         data_path="./data/data.txt", label_path="./data/label.txt")
