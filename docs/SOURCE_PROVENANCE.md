# Source provenance and compatibility notes

The recovery reference is the official repository commit
`20d8d7f91c8c5f409ef31081257676d655a27f29` from
<https://github.com/MikaZou/GLEEP>.

The independent implementation preserves the following published-code behavior only when
`--semantics published` is selected:

- `ResNet18()` uses the `[3, 4, 6, 3]` block layout and `ResNet34()` uses `[2, 2, 2, 2]`.
- CIFAR10 source checkpoints have a 100-output classifier even though only labels 0 through 9 are trained.
- The ImageNet score branch clusters the ImageNet classifier outputs but applies the CIFAR checkpoint classifier to the captured ImageNet features for LEEP.
- Each EXP2 task discards an initial seeded class permutation before `prepare_data()` draws the permutation that actually filters CIFAR100.
- `Retrain` freezes all parameters except the classifier and is therefore linear probing.
- The historical score is the average predicted probability, not the canonical average log-likelihood described in the manuscript.

The source repository also contains hard-coded loops, a duplicate `pets` entry, an unfixed GMM
random state, and a duplicate `-m` argparse option in `Retrain.py`. These are not copied into the
new command-line interface.

The original workspace contains several Python files made entirely of NUL bytes. Their paths and
the remaining syntax errors are listed by `python -m gleep_repro audit`. The new implementation
does not import those damaged files.
