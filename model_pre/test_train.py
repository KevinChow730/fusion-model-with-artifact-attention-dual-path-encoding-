import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np

from train import ViTacDataset  # 直接复用
from model import Trans
from process import Process

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    data_dir = "./data"
    label_dir = "./data"
    augment = False              # 过拟合测试时一般关闭增强
    window_size = 200
    batch_size = 10              # 一次性把 10 个样本都喂进去
    num_epochs = 200             # 训练多一点，看能否把 loss 压到很低
    seed = 42

    torch.manual_seed(seed)
    np.random.seed(seed)

    # 构造数据集
    processor = Process(model_path=None)
    full_dataset = ViTacDataset(
        data_dir,
        label_dir,
        processor=processor,
        augment=augment,
        window_size=window_size,
        step_size=200,
    )

    N = len(full_dataset)
    print(f"完整数据集大小: {N}")
    if N < 10:
        raise RuntimeError("数据集样本数不足 10，无法做过拟合小样本测试。")

    # 随机选 10 个索引
    idx = np.random.choice(N, size=10, replace=False)
    small_dataset = Subset(full_dataset, idx)
    print("小样本索引:", idx)

    loader = DataLoader(small_dataset, batch_size=batch_size, shuffle=True)

    # 构造模型
    model = Trans(input_len=window_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 只在这 10 个样本上训练与评估
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(small_dataset)

        # 也在同一批数据上评估一次（本质一样）
        model.eval()
        with torch.no_grad():
            eval_running = 0.0
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                eval_running += loss.item() * x.size(0)
        eval_loss = eval_running / len(small_dataset)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | eval_loss={eval_loss:.6f}")

    # 过拟合能力判断：train\_loss / eval\_loss 是否接近 0
    print("小样本过拟合测试完成。")


if __name__ == "__main__":
    main()
