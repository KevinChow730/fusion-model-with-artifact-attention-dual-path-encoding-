import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from train import ViTacDataset, train_vitac
from process import Process
from model import Trans

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # === 路径：这里必须是目录，不是单个 .txt 文件 ===
    data_dir = "./motion_train/data"          # 目录内放 data.txt（或多个 .txt）
    label_dir = "./motion_train/label"         # 目录内放 label.txt（或多个 .txt）
    pretrained_path = "./bestmodel/dynamic/dpa/model_1.pth"
    save_path = "./demo/model.pth"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    window_size = 200
    step_size = 10
    batch_size = 16
    num_epochs = 100
    lr = 1e-4
    seed = 42
    augment = False

    processor = Process(model_path=None)
    dataset = ViTacDataset(
        data_dir=data_dir,
        label_dir=label_dir,
        processor=processor,
        augment=augment,
        window_size=window_size,
        step_size=step_size,
    )

    if len(dataset) == 0:
        raise RuntimeError(
            "数据集为空：请确认 `data_dir`/`label_dir` 为包含 .txt 的目录，且目录下能匹配到成对的 `*.txt` 文件。"
        )

    N = len(dataset)
    val_len = max(1, int(0.2 * N))
    train_len = N - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = Trans(input_len=window_size).to(device)

    ckpt = torch.load(pretrained_path, map_location=device)
    model.load_state_dict(ckpt, strict=True)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_vitac(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=num_epochs,
        save_path=save_path,
        curve_path="./demo/loss_curve.png",
        curve_title="Finetune loss curves",
        log_txt_path="./demo/loss_log.txt",
        log_every=1,
    )

    print(f"微调完成，已输出最优权重到: {save_path}")


if __name__ == "__main__":
    main()