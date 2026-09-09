# GLEEP published-result verification

Profile: `published`  
Correlation: ordinary Kendall tau-b and Pearson correlation. Match tolerance: ±0.0015.

## Experiment 1: Tables 1 and 2

| Metric | Datasets | Kendall | Paper | Pearson | Paper | All cells |
| --- | --- | --- | --- | --- | --- | --- |
| LEEP | 11 | 0.544895 | 0.544 | 0.489779 | 0.489 | PASS |
| NLEEP | 11 | 0.438537 | 0.488 | 0.522347 | 0.531 | DIFF |
| LOGME | 11 | 0.488056 | 0.488 | 0.444450 | 0.444 | PASS |
| PAC | 11 | 0.143937 | 0.172 | 0.000308 | 0.045 | DIFF |
| SFDA | 11 | 0.597375 | 0.597 | 0.734311 | 0.734 | PASS |
| GLEEP | 11 | 0.581168 | 0.581 | 0.615408 | 0.615 | PASS |

The archived score files contain 10 models per complete dataset, while the manuscript says 11.
The table reports GLEEP Pearson 0.615; the prose reports 0.618.

## Experiment 2: Table 3

| Strategy | Model | Source | Metric | Pairs | Pearson | Paper | Kendall | Paper | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Finetune | ResNet18 | CIFAR10 | LEEP | 99 | 0.925463 | 0.925 | 0.933210 | 0.936 | DIFF |
| Finetune | ResNet18 | CIFAR10 | GLEEP | 99 | 0.954588 | 0.955 | 0.930324 | 0.931 | PASS |
| Finetune | ResNet18 | ImageNet | LEEP | 99 | 0.792511 | 0.793 | 0.943105 | 0.943 | PASS |
| Finetune | ResNet18 | ImageNet | GLEEP | 99 | 0.807393 | 0.807 | 0.938982 | 0.939 | PASS |
| Finetune | ResNet34 | CIFAR10 | LEEP | 99 | 0.902184 | 0.902 | 0.936289 | 0.933 | DIFF |
| Finetune | ResNet34 | CIFAR10 | GLEEP | 99 | 0.927811 | 0.928 | 0.931340 | 0.930 | PASS |
| Finetune | ResNet34 | ImageNet | LEEP | 99 | 0.789970 | 0.790 | 0.946403 | 0.946 | PASS |
| Finetune | ResNet34 | ImageNet | GLEEP | 99 | 0.809555 | 0.809 | 0.947227 | 0.947 | PASS |
| Retrain | ResNet18 | CIFAR10 | LEEP | 99 | 0.944350 | 0.944 | 0.980210 | 0.980 | PASS |
| Retrain | ResNet18 | CIFAR10 | GLEEP | 99 | 0.970002 | 0.970 | 0.976087 | 0.976 | PASS |
| Retrain | ResNet18 | ImageNet | LEEP | 99 | 0.921976 | 0.922 | 0.964956 | 0.965 | PASS |
| Retrain | ResNet18 | ImageNet | GLEEP | 64 | 0.959456 | 0.959 | 0.940476 | 0.940 | PASS |
| Retrain | ResNet34 | CIFAR10 | LEEP | 99 | 0.935519 | 0.936 | 0.977631 | 0.977 | PASS |
| Retrain | ResNet34 | CIFAR10 | GLEEP | 99 | 0.955321 | 0.955 | 0.972683 | 0.973 | PASS |
| Retrain | ResNet34 | ImageNet | LEEP | 99 | 0.898568 | 0.898 | 0.971552 | 0.971 | PASS |
| Retrain | ResNet34 | ImageNet | GLEEP | 99 | 0.913361 | 0.913 | 0.969491 | 0.969 | PASS |

| Metric | Average Pearson | Paper | Average Kendall | Paper |
| --- | --- | --- | --- | --- |
| LEEP | 0.888818 | 0.888 | 0.956669 | 0.956 |
| GLEEP | 0.912186 | 0.912 | 0.950826 | 0.951 |

### Detected anomalies

- Retrain|ResNet18|ImageNet|GLEEP contains 64 pairs instead of 99.
- The two CIFAR10 Finetune LEEP Kendall values match the opposite ResNet18/ResNet34 rows.
- The code iterates from 2 through 100 classes, so the normal count is 99 rather than 100.
- `Retrain` freezes every non-classifier parameter and is therefore linear probing.

## Verdict

PARTIAL REPRODUCTION: the four main Experiment 1 averages match, and 14/16 Experiment 2 rows match both published rounded coefficients. Known NLEEP/PAC, task-count, and manuscript inconsistencies remain visible.
