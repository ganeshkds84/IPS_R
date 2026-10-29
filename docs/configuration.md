# Configuration

Configuration is composed by [Hydra](https://hydra.cc) from `configs/` and
validated against the structured schema `ips_r.config.schema.AppConfig`.
Unknown keys and type mismatches are rejected when the config is composed.

## Structure

```
configs/
├── config.yaml              primary config + defaults list
├── paths/default.yaml       filesystem layout
├── logging/default.yaml     level, format, file logging
├── seed/default.yaml        global seed, determinism
├── experiment/              optional presets (# @package _global_)
│   └── example.yaml
├── dataset/none.yaml        ┐
├── retriever/none.yaml      │
├── candidate_generator/…    │  one group per component kind;
├── decision_policy/…        │  `none` = disabled (name: null)
├── reranker/…               │
├── generator/…              │
├── evaluator/…              │
└── pipeline/none.yaml       ┘
```

Every component node has the same shape:

```yaml
name: <registry name or null>
params: {}   # free-form, passed to the implementation's constructor
```

## Loading

```bash
uv run ips-r config                                   # print resolved config
uv run ips-r config experiment=example seed.value=3   # with overrides
```

```python
from ips_r.config import load_config, as_app_config

cfg = load_config(["seed.value=3"])     # DictConfig, fully resolved
typed = as_app_config(cfg)              # AppConfig dataclass instance
```

The loader uses Hydra's compose API, so Hydra does **not** change the working
directory or take over logging. Run directories are managed by `ips_r.experiment`.

## Override cheat-sheet

| Goal | Override |
| --- | --- |
| Change a value | `seed.value=7` |
| Select a group option | `retriever=my_retriever` |
| Apply a preset | `experiment=example` |
| Add a new component param | `+retriever.params.top_k=50` |
| Write the run to a fixed directory | `run.output_dir=/tmp/run1` |

## Environment variables

The `.env` file (see `.env.example`) is loaded at CLI start-up. Variables that
are already set in the shell take precedence.

| Variable | Effect |
| --- | --- |
| `IPS_R_ENV` | Deployment context label |
| `IPS_R_LOG_LEVEL`, `IPS_R_LOG_FORMAT` | Default logging (`logging.*`) |
| `IPS_R_SEED` | Default `seed.value` |
| `IPS_R_ROOT` | Project root (else auto-detected) |
| `IPS_R_CONFIG_DIR` | Config directory (else `<root>/configs`) |
| `IPS_R_DATA_DIR`, `IPS_R_CACHE_DIR`, `IPS_R_MODELS_DIR`, `IPS_R_OUTPUT_DIR` | Redirect working directories |

Precedence, from highest to lowest: CLI override → shell env → `.env` → YAML default.
