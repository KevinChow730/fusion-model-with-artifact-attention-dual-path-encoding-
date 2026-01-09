import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import numpy as np
import other_model as om


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VADDataset(Dataset):
    def __init__(self, data_dir, label_dir, augment=False):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.data_files = sorted(self.data_dir.glob("*.txt"))  # 改为txt文件
        self.label_files = sorted(self.label_dir.glob("*.txt"))
        self.augment = augment

        self.data, self.labels = self.process_data()

    def process_data(self):
        data = []
        labels = []

        for data_file, label_file in zip(self.data_files, self.label_files):
            # 读取整个数据文件
            file_data = []
            with open(data_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    values = [float(x) for x in line.strip().split()]
                    if len(values) == 2:
                        file_data.append(values)

            # 读取标签文件（每行2个值：最大值 最小值）
            file_labels = []
            with open(label_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        label_values = [float(x) for x in line.strip().split()]
                        if len(label_values) == 2:
                            file_labels.append(label_values)

            # 滑动窗口处理
            window_size = 128
            step_size = 1

            for i in range(0, len(file_data) - window_size + 1, step_size):
                # 确保标签索引不超出范围
                if i >= len(file_labels):
                    break

                window_data = file_data[i:i + window_size]

                col1 = [row[0] for row in window_data]
                col2 = [row[1] for row in window_data]

                token1 = np.array(col1, dtype=np.float32)
                token2 = np.array(col2, dtype=np.float32)

                if self.augment:
                    token1 = self.add_noise(token1)
                    token2 = self.add_noise(token2)

                feature_array = np.stack([token1, token2], axis=0)
                data.append(feature_array)

                # 添加2个标签值 [最大值, 最小值]
                labels.append(file_labels[i])

        return np.array(data, dtype=np.float32), np.array(labels, dtype=np.float32)

    def add_noise(self, data):
        noise = np.random.normal(0, 0.01, size=data.shape)  # 高斯噪声
        return data + noise

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.from_numpy(self.data[idx]).float()  # [2, 64]
        label = torch.from_numpy(self.labels[idx]).float()  # [2] - 包含最大值和最小值
        return sample, label


def train_vad(model, train_loader, val_loader, criterion, optimizer, num_epochs=150, save_path="./model/other.pth"):
    Path("./model").mkdir(exist_ok=True)  # 创建保存目录
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            # outputs: [batch_size, 2], labels: [batch_size, 2]
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}')

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_labels.append(labels.cpu().numpy())
                all_preds.append(outputs.cpu().numpy())

        # 合并所有批次
        all_labels = np.concatenate(all_labels, axis=0)  # [N, 2]
        all_preds = np.concatenate(all_preds, axis=0)    # [N, 2]

        # 分别计算两个输出的指标
        val_loss_avg = val_loss / len(val_loader)
        mse_max = np.mean((all_labels[:, 0] - all_preds[:, 0]) ** 2)  # 最大值MSE
        mse_min = np.mean((all_labels[:, 1] - all_preds[:, 1]) ** 2)  # 最小值MSE
        mae_max = np.mean(np.abs(all_labels[:, 0] - all_preds[:, 0]))  # 最大值MAE
        mae_min = np.mean(np.abs(all_labels[:, 1] - all_preds[:, 1]))  # 最小值MAE

        print(f'Validation Loss: {val_loss_avg:.4f}')
        print(f'最大值 - MSE: {mse_max:.4f}, MAE: {mae_max:.4f}')
        print(f'最小值 - MSE: {mse_min:.4f}, MAE: {mae_min:.4f}')

        # 保存最佳模型
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'mse_max': mse_max,
                'mse_min': mse_min,
                'mae_max': mae_max,
                'mae_min': mae_min
            }, save_path)
            print(f"✓ 保存最佳模型到 {save_path} (验证损失: {val_loss_avg:.4f})")

        print('-' * 50)

    print(f"\n训练完成！最佳验证损失: {best_val_loss:.4f}")
    print(f"最佳模型已保存到: {save_path}")


if __name__ == "__main__":
    data_dir = "./data/3"
    label_dir = "./label/3"
    augment = True
    dataset = VADDataset(data_dir, label_dir, augment=augment)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    model = om.EncoderOnlyTransformer().to(device)  # 移动到设备
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print(f"使用设备: {device}")
    print(f"数据集大小: {len(dataset)}")
    print(f"训练集大小: {train_size}, 验证集大小: {val_size}")

    train_vad(model, train_loader, val_loader, criterion, optimizer, num_epochs=500)
    torch.save(model.state_dict(), "./model/other.pth")
