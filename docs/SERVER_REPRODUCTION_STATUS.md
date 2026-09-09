# GLEEP server reproduction status

Snapshot date: 2026-09-05 (Asia/Shanghai)  
Server workspace: private; public artifacts are identified by relative path and SHA256.

## Runtime and data

- Scheduler: Slurm; GPU jobs run on `acd_u` or `debug` rather than the login node.
- Compatible shared runtime used for the current runs: Python 3.11.7, PyTorch 2.1.2+cu121, TorchVision 0.16.2+cu121, NumPy 1.26.3, SciPy 1.11.4, scikit-learn 1.2.2.
- The exact environment in `environment.yml` could not be downloaded because the login node could not reach the Conda channels.
- CIFAR-100 archive MD5: `eb9058c3a382ffc7106e4002c42a8d85`.
- Extracted CIFAR-100 file MD5 values: `train=16019d7e3df5f24257cddd939b257f8d`, `test=f0ef6b0ae62326f3e7ffdfab6717acfc`, `meta=7973b15100ade9c7d40fb424638fde48`.
- Eight server regression tests pass.

## Reproduction fixes discovered on the server

1. Historical checkpoints are deserialized on CPU before the complete model is moved to CUDA. This avoids a legacy-storage device mismatch in newer PyTorch.
2. Published EXP2 semantics reproduce the official script's discarded first class permutation. With seed 42, the actual $k=2$ classes are `[53, 64]`, not `[42, 96]`.
3. EXP2 scores and a progress manifest are now written after every task so long jobs retain completed work.
4. Repacking excludes deployed ZIP files and `sample_workspace`, keeping the minimal package below 150 MiB.

## Completed checks

- Published JSON verification: the four principal EXP1 averages match; 14 of 16 EXP2 rows match both rounded paper coefficients.
- EXP1 Flowers cached-logit check: MobileNetV2 and ResNet34 LEEP absolute errors are about $9.67\times10^{-10}$ and $2.10\times10^{-9}$; deterministic GLEEP repeats exactly.
- EXP2 synthetic CUDA smoke: eight endpoint records cover ResNet18/34, CIFAR10 $k=2,100$, and ImageNet $k=2,10$; two runs produced byte-identical JSON and all scores are finite.
- EXP2 real-data endpoint checks completed for both models and both sources.

## Full deterministic EXP2 score recomputation completed

All score runs below contain 99 tasks and use seed 42, `random_state=0`, full-covariance GMM, published implementation semantics, the legacy average-probability formula, and historical downstream accuracies. The one 64-pair row is limited by its historical accuracy JSON rather than the recomputed scores.

| Model | Source | Accuracy branch | Metric | Pearson | Kendall tau-b |
| --- | --- | --- | --- | ---: | ---: |
| ResNet18 | CIFAR10 | Finetune | LEEP | 0.925461 | 0.933210 |
| ResNet18 | CIFAR10 | Retrain | LEEP | 0.944349 | 0.980210 |
| ResNet18 | CIFAR10 | Finetune | GLEEP | 0.954829 | 0.934859 |
| ResNet18 | CIFAR10 | Retrain | GLEEP | 0.969964 | 0.971965 |
| ResNet34 | CIFAR10 | Finetune | LEEP | 0.902188 | 0.936289 |
| ResNet34 | CIFAR10 | Retrain | LEEP | 0.935522 | 0.977631 |
| ResNet34 | CIFAR10 | Finetune | GLEEP | 0.926614 | 0.926392 |
| ResNet34 | CIFAR10 | Retrain | GLEEP | 0.955675 | 0.972271 |
| ResNet18 | ImageNet | Finetune | LEEP | 0.792511 | 0.943105 |
| ResNet18 | ImageNet | Retrain | LEEP | 0.921976 | 0.964956 |
| ResNet18 | ImageNet | Finetune | GLEEP | 0.811117 | 0.944341 |
| ResNet18 | ImageNet | Retrain (64 pairs) | GLEEP | 0.959956 | 0.944444 |
| ResNet34 | ImageNet | Finetune | LEEP | 0.789975 | 0.946403 |
| ResNet34 | ImageNet | Retrain | LEEP | 0.898572 | 0.971552 |
| ResNet34 | ImageNet | Finetune | GLEEP | 0.797251 | 0.941868 |
| ResNet34 | ImageNet | Retrain | GLEEP | 0.903918 | 0.965368 |

For the two full CIFAR10-source runs, deterministic LEEP differs from the historical score files by less than $1.1\times10^{-5}$ at every task. Deterministic GLEEP differs because the historical GMM did not record a random state; the mean absolute task-level differences are 0.003710 (ResNet18) and 0.003529 (ResNet34).

## Persistent EXP2 intermediate cache

- Full ImageNet score jobs `593606` and `593657` completed successfully in 39:47 and 42:04.
- Cache extraction job `595051` completed successfully in 22 seconds.
- The cache is under `results/intermediate/exp2_logits/published/` and occupies about 178 MiB.
- Each of the four model/source groups contains 10,000 labels, 512-dimensional classifier-input features, clustering logits, prediction logits, and a provenance manifest.
- All 16 arrays are finite and their current SHA256 values match the manifests.
- Every CIFAR100 class contains 100 cached test samples, and all 99 task subsets reconstruct the expected $100k$ samples.
- Cached-score validation jobs `595060`, `595062`, and `595065` completed successfully. All 396 cached LEEP task scores were reproduced in 19 seconds; versus direct per-task forward, the maximum absolute delta is $1.71\times10^{-5}$ and the mean is $1.41\times10^{-7}$. The four $k=2$ GLEEP checks have the same maximum tolerance. These tiny differences come from changing the inference batch layout, while every $k=100$ LEEP endpoint is exactly equal.
- Future score runs can bypass source-model forward with `--mode score --score-input cache`.

Completed manifests and Slurm logs are under `results/` and `reports/` respectively. Published historical JSON remains unchanged under `historical/`. The updated minimal package is 115.05 MiB with SHA256 `5898abc3cdf3103353dfc37512e9b1ea474e15289772cd3b7654bed138f00a29`.
