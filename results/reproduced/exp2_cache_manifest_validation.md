# EXP2 server-cache manifest validation

The four manifests were downloaded together with the server-generated cache. Every local array was re-hashed and checked against its manifest.

| Model | Source | Array | Bytes | SHA256 | Shape | Dtype | Finite |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet18 | CIFAR10 | labels | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | CIFAR10 | features | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | CIFAR10 | clustering_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | CIFAR10 | prediction_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | ImageNet | labels | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | ImageNet | features | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | ImageNet | clustering_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet18 | ImageNet | prediction_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | CIFAR10 | labels | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | CIFAR10 | features | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | CIFAR10 | clustering_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | CIFAR10 | prediction_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | ImageNet | labels | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | ImageNet | features | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | ImageNet | clustering_logits | PASS | PASS | PASS | PASS | PASS |
| ResNet34 | ImageNet | prediction_logits | PASS | PASS | PASS | PASS | PASS |

All four label hashes identical: **PASS**

Verdict: **PASS**
