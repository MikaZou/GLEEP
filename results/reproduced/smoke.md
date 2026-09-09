# GLEEP smoke test

## EXP1 cached logits

| Dataset | Model | LEEP | Historical | Delta | GLEEP deterministic |
| --- | --- | --- | --- | --- | --- |
| flowers | mobilenet_v2 | 0.063496690927 | 0.063496689960 | 9.671e-10 | True |
| flowers | resnet34 | 0.059276656599 | 0.059276654498 | 2.101e-09 | True |

GLEEP smoke runs on a real 10-class/32-logit slice with diagonal covariance so it finishes reliably; it is not compared with the full paper score.
