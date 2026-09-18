"""Единый контракт модели для обучения и приложения."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# ImageFolder сортирует имена папок по алфавиту. Не меняйте порядок.
CLASSES = ("healthy", "leaf_rust", "septoria", "yellow_rust")
MODEL_PATH = ROOT / "model" / "wheat_model.pth"


def inference_transform():
    """Стабильное преобразование для val, test и инференса."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def build_model(pretrained: bool = False):
    """MobileNetV3 Small с четырьмя выходами; приложение не скачивает веса."""
    from torch import nn
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
    model = mobilenet_v3_small(
        weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    return model
