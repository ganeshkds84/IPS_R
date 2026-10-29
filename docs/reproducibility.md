# Reproducibility

## Seeds

`ips_r.core.seed.seed_everything(seed, deterministic=False)` seeds:

- Python `random`
- NumPy's legacy global RNG
- PyTorch CPU and CUDA, if torch is installed; with `deterministic=True` it also
  enables deterministic algorithms and cuDNN determinism
- `PYTHONHASHSEED`, which affects **child** processes only

Prefer explicit randomness over global state:

```python
from ips_r.core.seed import derive_seed, make_rng

rng = make_rng(derive_seed(cfg.seed.value, "retriever", fold))
```

`derive_seed` is a SHA-256 based derivation. It does not depend on Python's hash
randomisation, so child seeds are stable across processes and machines.

## Run directories

Each `ips-r run` writes `outputs/<experiment>/<YYYYmmdd-HHMMSS>-<hex>/`:

| File | Contents |
| --- | --- |
| `config.yaml` | Fully resolved config; re-run with `ips-r run` plus the same overrides |
| `metadata.json` | `status` (`running` / `completed` / `failed`), timings, git commit (`-dirty` when there are uncommitted changes), Python and platform, argv, seed state, resolved paths, error message on failure |
| `run.log` | JSON-lines log of the run |
| `results.json` | Pipeline summary |

## Dependencies

`uv.lock` pins every transitive dependency. CI and Docker install with
`uv sync --frozen`, so they fail if the lock file is out of date.
