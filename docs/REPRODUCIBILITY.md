# Reproducibility protocol

## Scope

This repository covers the two main experiments and Tables 1--3. It does not claim to
reproduce the ablation study, timing study, or Figures 4--5. The manuscript source is
kept outside the public code repository.

## Profiles

`published` is an immutable compatibility profile. It retains the archived model
order, 99-task EXP2 construction, historical mean-probability LEEP behavior, and
original ImageNet branch semantics. Only this profile is compared with paper tables.

`corrected` is reserved for deterministic or formula-corrected experiments. Its
outputs use separate names and must never replace published JSON.

## Artifact policy

Git contains code, configuration, small JSON results, reports, and an artifact index.
Arrays, datasets, environments, and checkpoints are excluded. Each retained binary
has a relative path, byte count, SHA256, role, storage state, and rebuild command in
`artifacts/index.json`.

The retained local set is:

- 121 EXP1 logits arrays and 121 label arrays;
- EXP2 features, clustering logits, prediction logits, labels, and manifests for
  ResNet18/34 with CIFAR10/ImageNet source models;
- ResNet18 and ResNet34 CIFAR10 source checkpoints.

EXP1 features are not retained because GLEEP and both LEEP variants can operate from
logits. EXP2 downstream checkpoints are not retained because their best accuracies are
captured in historical JSON and they can be regenerated one task at a time.

## Verification gates

Before release or cleanup:

1. run unit tests;
2. run `python -m gleep_repro verify --profile published`;
3. run `python -m gleep_repro artifacts audit` where artifacts are available;
4. reload every retained NumPy array with pickle disabled and check shape, dtype, and
   finite values;
5. run EXP1 Flowers cached checks for MobileNetV2 and ResNet34;
6. run EXP2 cached checks at class counts 2, 10, and 100 for all four sources;
7. ensure Git tracks no model/data arrays, file over 50 MiB, credential, or private
   absolute workspace path;
8. run JSON-only verification from a clean clone.

## Interpretation

Reports use ordinary Kendall tau-b and Pearson correlation, not weighted Kendall.
Every result identifies its source and actual task count. A changed seed, missing
record, or corrected formula is an experimental change and must be named explicitly.
