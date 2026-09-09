from __future__ import annotations


def build_cifar_resnet(model_name: str, *, num_classes: int = 100, semantics: str = "published"):
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
    except Exception as exc:
        raise RuntimeError("PyTorch is required; install environment.yml first") from exc

    class BasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_channels, out_channels, stride=1):
            super().__init__()
            self.conv1 = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels),
                )

        def forward(self, values):
            output = functional.relu(self.bn1(self.conv1(values)))
            output = self.bn2(self.conv2(output))
            output += self.shortcut(values)
            return functional.relu(output)

    class CifarResNet(nn.Module):
        def __init__(self, blocks):
            super().__init__()
            self.in_channels = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.layer1 = self._make_layer(64, blocks[0], stride=1)
            self.layer2 = self._make_layer(128, blocks[1], stride=2)
            self.layer3 = self._make_layer(256, blocks[2], stride=2)
            self.layer4 = self._make_layer(512, blocks[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)

        def _make_layer(self, out_channels, block_count, stride):
            strides = [stride] + [1] * (block_count - 1)
            layers = []
            for item_stride in strides:
                layers.append(BasicBlock(self.in_channels, out_channels, item_stride))
                self.in_channels = out_channels
            return nn.Sequential(*layers)

        def forward(self, values):
            values = functional.relu(self.bn1(self.conv1(values)))
            values = self.layer1(values)
            values = self.layer2(values)
            values = self.layer3(values)
            values = self.layer4(values)
            values = self.avgpool(values)
            values = torch.flatten(values, 1)
            return self.fc(values)

    if model_name not in {"ResNet18", "ResNet34"}:
        raise ValueError(f"unsupported CIFAR model: {model_name}")
    if semantics == "published":
        # The official repository accidentally swaps the two depth definitions.
        blocks = [3, 4, 6, 3] if model_name == "ResNet18" else [2, 2, 2, 2]
    elif semantics == "corrected":
        blocks = [2, 2, 2, 2] if model_name == "ResNet18" else [3, 4, 6, 3]
    else:
        raise ValueError(f"unknown model semantics: {semantics}")
    return CifarResNet(blocks)


def build_imagenet_resnet(model_name: str, *, pretrained: bool = True):
    try:
        from torchvision import models
    except Exception as exc:
        raise RuntimeError("TorchVision is required; install environment.yml first") from exc
    if model_name == "ResNet18":
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet18(weights=weights)
    if model_name == "ResNet34":
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        return models.resnet34(weights=weights)
    raise ValueError(f"unsupported ImageNet model: {model_name}")

