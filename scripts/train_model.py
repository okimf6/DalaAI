"""Обучение MobileNetV3 Small: train/val/test с проверкой единого порядка классов."""
import argparse
import random
import sys
from pathlib import Path

# Позволяет запускать python scripts/train_model.py из папки проекта.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from utils.model_config import CLASSES, MODEL_PATH, build_model, inference_transform


def set_seed(seed: int) -> None:
    """Фиксирует генераторы; CUDA использует детерминированные алгоритмы по возможности."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loaders(data: Path, batch_size: int, seed: int) -> dict[str, DataLoader]:
    """Проверяет папки ImageFolder и добавляет аугментации только к train."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    loaders = {}
    for split in ("train", "val", "test"):
        folder = data / split
        if not folder.is_dir():
            raise ValueError(f"Не найдена папка датасета: {folder}")
        dataset = datasets.ImageFolder(folder, transform=train_transform if split == "train" else inference_transform())
        if tuple(dataset.classes) != CLASSES:
            raise ValueError(f"Классы {split} должны быть строго {list(CLASSES)}; получено {dataset.classes}")
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                                   num_workers=0, generator=torch.Generator().manual_seed(seed))
    return loaders


def run_epoch(model: nn.Module, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """Один проход: средняя cross entropy и accuracy по всем объектам."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss):
                raise ValueError("Функция потерь стала некорректной. Проверьте данные и learning rate.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            count += targets.size(0)
    if count == 0:
        raise ValueError("Выборка не содержит изображений.")
    return total_loss / count, correct / count


def train(args: argparse.Namespace) -> dict:
    """Обучает модель, сохраняет лучший checkpoint и оценивает его на test."""
    if args.epochs < 1 or args.batch_size < 2 or args.lr <= 0 or args.patience < 1:
        raise ValueError("Эпохи и patience должны быть >= 1, batch size >= 2, learning rate > 0.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    loaders = make_loaders(args.data, args.batch_size, args.seed)
    model = build_model(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0
    stale = 0
    print(f"Устройство: {device}. Порядок классов: {list(CLASSES)}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], device, optimizer)
        val_loss, val_accuracy = run_epoch(model, loaders["val"], device)
        print(f"Эпоха {epoch:02d}/{args.epochs}: train loss={train_loss:.4f}, accuracy={train_accuracy:.2%}; "
              f"val loss={val_loss:.4f}, accuracy={val_accuracy:.2%}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            stale = 0
            checkpoint = {
                "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "classes": list(CLASSES), "best_val_accuracy": best_accuracy,
                "epoch": epoch, "architecture": "mobilenet_v3_small",
                "training_params": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            }
            temporary = args.output.with_suffix(".tmp")
            torch.save(checkpoint, temporary)
            temporary.replace(args.output)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Ранняя остановка: {args.patience} эпох без улучшения.")
                break
    checkpoint = torch.load(args.output, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = run_epoch(model, loaders["test"], device)
    checkpoint["test_accuracy"] = test_accuracy
    checkpoint["test_loss"] = test_loss
    temporary = args.output.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(args.output)
    print(f"Лучшая val accuracy: {best_accuracy:.2%}. Test accuracy: {test_accuracy:.2%}, loss: {test_loss:.4f}")
    print(f"Модель сохранена: {args.output}")
    return checkpoint


def parse_args() -> argparse.Namespace:
    """Параметры обучения из командной строки."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Папка с train, val и test")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        train(parse_args())
    except KeyboardInterrupt:
        print("Обучение прервано пользователем. Лучший сохранённый checkpoint остаётся на диске.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"Не удалось завершить обучение: {exc}. Проверьте структуру датасета, изображения и доступ к весам ImageNet.", file=sys.stderr)
        sys.exit(1)
