# GLEEP reproducible-v2

This branch rebuilds GLEEP on official commit
`20d8d7f91c8c5f409ef31081257676d655a27f29`. The original `EXP1/` and `EXP2/`
directories remain for provenance; `gleep_repro/` provides deterministic, auditable
entry points for the paper's main experiments.

## Quick start

```bash
conda env create -f environment.yml
conda activate gleep-repro
python -m gleep_repro verify --profile published
python -m gleep_repro audit
python -m gleep_repro artifacts audit
```

`verify` recomputes Tables 1--3 from tracked historical JSON without datasets,
weights, or cached arrays. The artifact audit additionally validates local,
Git-ignored binaries against `artifacts/index.json`.

## Layout

```text
EXP1/                         original Experiment 1 source
EXP2/                         original Experiment 2 source
gleep_repro/                  deterministic reproduction package
configs/                      published and corrected protocols
results/published/            historical score and accuracy JSON
results/reproduced/           regenerated tables and audit evidence
results/research/             current method research outputs
research/negative_results/    documented unsuccessful directions
artifacts/index.json          hashes and provenance for ignored binaries
docs/                         reproduction methodology and status
tests/                        unit tests
```

## Reproduction levels

1. **JSON-only:** clone and run `verify` to recompute reported correlations.
2. **Cached scoring:** restore indexed artifacts, then recompute GLEEP/LEEP or test
   a new metric without another source-model forward pass.
3. **End to end:** download public data and ImageNet weights, restore the CIFAR10
   source checkpoints, regenerate caches, and run the complete protocol.

```bash
python -m gleep_repro run exp1 --source cached --metrics gleep leep
python -m gleep_repro run exp2 --mode score --score-input cache
python -m gleep_repro run exp2 --mode train --retain-checkpoints none
python -m gleep_repro artifacts fetch --profile exp2
```

Artifact download URIs remain unset until a stable Hugging Face or cloud release is
published. Hashes, sizes, shapes, dtypes, roles, and rebuild commands are already
recorded, so later storage migration will not change artifact identity.

## Published protocol and known limitations

- EXP1's archived tables contain 10 models and 11 target datasets. InceptionV3 is an
  extra experiment and is not mixed into the historical table.
- The official script lists `pets` twice; the reproducible configuration lists it once.
- EXP2 evaluates class counts 2 through 100 inclusive, which is 99 tasks.
- Historical `Retrain` freezes the backbone and is therefore linear probing.
- `Retrain/ResNet18/ImageNet/GLEEP` contains only 64 historical task pairs.
- The original GMM seed was not fixed. Corrected runs use `random_state=0` and never
  overwrite historical output.
- The manuscript analyzes mean log probability, while historical code averages
  probabilities. `published` retains the historical behavior; canonical log-LEEP
  results use separate names.
- Two CIFAR10 LEEP Kendall entries in Table 3 appear exchanged between ResNet18 and
  ResNet34. Verification reports this rather than altering the record.
- The official EXP1 ImageNet dataset stub is syntactically incomplete. It is a
  provenance file and is not used by the main benchmark.
- UPR-v1 and relational-v1 are failed explorations, not mature proposed methods.

The evidence supports a **partial historical reproduction**: the main EXP1
GLEEP/LEEP/LogME/SFDA averages and most EXP2 correlations are recoverable, while known
data and manuscript inconsistencies remain explicit. See
[`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) and
[`results/reproduced/`](results/reproduced/).

## Citation and license

The original project is [MikaZou/GLEEP](https://github.com/MikaZou/GLEEP). This branch
keeps its MIT [`LICENSE`](LICENSE). Cite the GLEEP paper when using its method or data.
