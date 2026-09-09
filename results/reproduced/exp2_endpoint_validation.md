# EXP2 cached endpoint validation

Pinned environment: Python 3.10, NumPy 1.24.4, SciPy 1.10.1, scikit-learn 1.2.2. Seed 42, GMM random state 0, full covariance.
Same-environment repeats use tolerance 1e-12; comparison with the archived full run uses a declared 5e-5 cross-machine tolerance.

| Model | Source | k | Metric | Score | Repeat delta | Full-run delta |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| ResNet18 | CIFAR10 | 2 | GLEEP | 0.626477604797 | 0.000e+00 | 4.957e-07 |
| ResNet18 | CIFAR10 | 100 | GLEEP | 0.045857437622 | 0.000e+00 | 0.000e+00 |
| ResNet18 | CIFAR10 | 2 | LEEP | 0.550595180675 | 0.000e+00 | 4.170e-06 |
| ResNet18 | CIFAR10 | 100 | LEEP | 0.022687850580 | 0.000e+00 | 0.000e+00 |
| ResNet18 | ImageNet | 2 | GLEEP | 0.546637139234 | 0.000e+00 | 2.311e-08 |
| ResNet18 | ImageNet | 10 | GLEEP | 0.123522301293 | 0.000e+00 | 8.457e-09 |
| ResNet18 | ImageNet | 2 | LEEP | 0.542497829769 | 0.000e+00 | 1.388e-06 |
| ResNet18 | ImageNet | 10 | LEEP | 0.107456103383 | 0.000e+00 | 3.848e-08 |
| ResNet34 | CIFAR10 | 2 | GLEEP | 0.658941869499 | 0.000e+00 | 3.865e-06 |
| ResNet34 | CIFAR10 | 100 | GLEEP | 0.043077305306 | 0.000e+00 | 0.000e+00 |
| ResNet34 | CIFAR10 | 2 | LEEP | 0.585020484850 | 0.000e+00 | 2.906e-06 |
| ResNet34 | CIFAR10 | 100 | LEEP | 0.022261949066 | 0.000e+00 | 0.000e+00 |
| ResNet34 | ImageNet | 2 | GLEEP | 0.756909879921 | 0.000e+00 | 1.381e-05 |
| ResNet34 | ImageNet | 10 | GLEEP | 0.163522125350 | 0.000e+00 | 3.103e-07 |
| ResNet34 | ImageNet | 2 | LEEP | 0.764825149649 | 0.000e+00 | 1.703e-05 |
| ResNet34 | ImageNet | 10 | LEEP | 0.145550364829 | 0.000e+00 | 5.881e-07 |

Verdict: **PASS**
