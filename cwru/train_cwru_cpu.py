"""
基于 CPU 的 CWRU 轴承故障诊断训练脚本。

- 仅使用 CPU 进行训练，便于在没有 NVIDIA GPU 的环境中运行。
- 示例使用 4 个故障类别：
  - 正常：97.mat
  - 内圈故障：105.mat
  - 滚动体故障：118.mat
  - 外圈故障：130.mat

本脚本会将长序列振动信号切分成固定长度的窗口，以便训练一维卷积网络。
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch import nn
from torch.utils.data import DataLoader, Dataset

# 全局仅使用 CPU
device = torch.device("cpu")


class CWRUDataset(Dataset):
    """简单的 CWRU 数据集封装。

    - 读取 .mat 文件中的振动信号（默认使用带 `DE_time` 的键）。
    - 将长序列按滑动窗口切分成若干样本，节省内存并增加样本数量。
    """

    def __init__(
        self,
        root_dir: str,
        file_labels: Dict[str, int],
        window_size: int = 2048,
        step_size: int = 1024,
    ) -> None:
        self.samples: List[Tuple[np.ndarray, int]] = []
        for file_name, label in file_labels.items():
            file_path = os.path.join(root_dir, file_name)
            signal = self._load_signal(file_path)
            windows = self._window_signal(signal, window_size, step_size)
            self.samples.extend((win, label) for win in windows)

    @staticmethod
    def _load_signal(file_path: str) -> np.ndarray:
        """加载单个 .mat 文件并返回一维信号。"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"未找到数据文件: {file_path}")
        mat_data = loadmat(file_path)
        # CWRU 数据通常包含 `*_DE_time` 键，找到即可。
        data_key = next((k for k in mat_data.keys() if "DE_time" in k), None)
        if data_key is None:
            raise KeyError(f"文件中未找到 DE_time 键: {file_path}")
        signal = mat_data[data_key].squeeze()
        return signal.astype(np.float32)

    @staticmethod
    def _window_signal(signal: np.ndarray, window_size: int, step_size: int) -> List[np.ndarray]:
        """将长信号按窗口切分。"""
        windows: List[np.ndarray] = []
        for start in range(0, len(signal) - window_size + 1, step_size):
            end = start + window_size
            windows.append(signal[start:end])
        return windows

    def __len__(self) -> int:  # noqa: D401 - 简单长度返回
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        window, label = self.samples[idx]
        # 增加通道维度，形成 [C, L]
        tensor = torch.from_numpy(window).unsqueeze(0)
        return tensor, label


class SimpleCNN(nn.Module):
    """轻量级一维卷积分类器，适合 CPU 训练。"""

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 - 简单 forward
        return self.classifier(self.features(x))


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """单轮训练循环，返回平均损失。"""
    model.train()
    total_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(dataloader.dataset)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
    """在验证集上评估损失与准确率。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    return avg_loss, accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU 版 CWRU 故障诊断训练脚本")
    parser.add_argument("--data_dir", type=str, default="./data", help="包含 .mat 文件的数据目录")
    parser.add_argument("--batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=10, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--window_size", type=int, default=2048, help="滑动窗口长度")
    parser.add_argument("--step_size", type=int, default=1024, help="滑动窗口步长")
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="训练集比例，剩余用于验证",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 固定随机种子，便于复现
    torch.manual_seed(42)
    np.random.seed(42)

    # 数据文件与类别映射，用户可自行修改文件名
    file_labels = {
        "97.mat": 0,  # 正常
        "105.mat": 1,  # 内圈故障
        "118.mat": 2,  # 滚动体故障
        "130.mat": 3,  # 外圈故障
    }

    # 加载完整数据集
    dataset = CWRUDataset(
        root_dir=args.data_dir,
        file_labels=file_labels,
        window_size=args.window_size,
        step_size=args.step_size,
    )

    # 划分训练与验证集
    total_size = len(dataset)
    train_size = int(total_size * args.train_ratio)
    val_size = total_size - train_size
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    model = SimpleCNN(num_classes=len(file_labels)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 训练主循环
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )

    # 最终保存模型参数
    os.makedirs("./checkpoints", exist_ok=True)
    save_path = os.path.join("./checkpoints", "cwru_cnn_cpu.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
