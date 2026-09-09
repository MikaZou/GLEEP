# Binary artifacts

`cache/` and `checkpoints/` are local, Git-ignored storage. Their identities are
tracked in [`index.json`](index.json).

```bash
python -m gleep_repro artifacts audit
python -m gleep_repro artifacts fetch --profile exp2
```

Profiles are `exp1`, `exp2`, `source-checkpoints`, and `all`. Fetching returns a clear
error while `remote_uri` is null. After a stable external release, add URLs without
changing hashes or relative paths. Never commit arrays, checkpoints, datasets,
archives, or credentials.
