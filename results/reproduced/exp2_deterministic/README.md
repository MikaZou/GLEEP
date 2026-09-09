# Deterministic EXP2 cached-score evidence

These eight score maps were produced from the indexed EXP2 cache using the published
task construction, seed 42, GMM `random_state=0`, full covariance, and the historical
mean-probability formula. Each JSON contains all 99 class-count tasks from 2 through
100.

The maps cover ResNet18/34 with CIFAR10/ImageNet source models and both GLEEP and LEEP.
They are preserved separately from `results/published/`: deterministic recomputation
must not overwrite the original unseeded paper artifacts.

Representative endpoint checks use CIFAR10 tasks 2 and 100 and ImageNet tasks 2 and
10. The complete maps additionally retain task 100 for both ImageNet sources, so all
requested endpoints remain auditable without repeating the expensive GMM fit.
