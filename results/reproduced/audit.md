# GLEEP workspace audit

Workspace: `.`  
Official source: `20d8d7f91c8c5f409ef31081257676d655a27f29`

## Storage

| Area | Files | Size |
| --- | --- | --- |
| EXP1 feature cache | 0 | 0.00 B |
| EXP1 logits cache | 121 | 7.25 GiB |
| EXP1 labels | 121 | 14.87 MiB |
| EXP2 checkpoints | 0 | 0.00 B |
| EXP2 result JSON | 32 | 78.02 KiB |

## Python source health

No source corruption was detected.

### Official legacy source notes

| Path | Note |
| --- | --- |
| EXP1/datasets/imagenet.py | Official unused ImageNet dataset stub is syntactically incomplete. |
| EXP2/Ablation/test1.py | Official out-of-scope ablation placeholder is empty. |

## EXP2 checkpoint coverage

| Strategy | Model | Source | Checkpoints | Missing |
| --- | --- | --- | --- | --- |
| Finetune | ResNet18 | CIFAR10 | 0 | 99 |
| Finetune | ResNet18 | ImageNet | 0 | 99 |
| Finetune | ResNet34 | CIFAR10 | 0 | 99 |
| Finetune | ResNet34 | ImageNet | 0 | 99 |
| Retrain | ResNet18 | CIFAR10 | 0 | 99 |
| Retrain | ResNet18 | ImageNet | 0 | 99 |
| Retrain | ResNet34 | CIFAR10 | 0 | 99 |
| Retrain | ResNet34 | ImageNet | 0 | 99 |

## Runtime EXP2 data health

| Path | Exists | Size | First 4 KiB all NUL |
| --- | --- | --- | --- |
| artifacts/cache/data/cifar-100-python/train | False | 0.00 B | None |
| artifacts/cache/data/cifar-100-python/test | False | 0.00 B | None |
| artifacts/cache/data/cifar-100-python/meta | False | 0.00 B | None |
| artifacts/cache/data/cifar-100-python.tar.gz | False | 0.00 B | None |

## Runtime

- Python: `3.10.19 | packaged by conda-forge | (main, Oct 22 2025, 22:23:22) [MSC v.1944 64 bit (AMD64)]`
- Torch/NumPy bridge: `OK`
- GPU: `NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB, 580.97`
- cuDNN convolution: `OK`
- Native CUDA convolution: `OK`
- cuBLAS matrix multiply: `OK`

Official `EXP1`/`EXP2` remain provenance sources. `gleep_repro` is the deterministic runnable implementation.
