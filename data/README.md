# Data

Git ignores the contents of this folder, apart from the directory skeleton.

| Folder | Contents |
| --- | --- |
| `raw/` | Original, immutable source data |
| `external/` | Third-party resources (e.g. pre-built indices, word lists) |
| `interim/` | Intermediate transformations |
| `processed/` | Final, model-ready datasets |

To relocate it, for example to a scratch disk, set `IPS_R_DATA_DIR`.

## Default source locations

`configs/dataset/*.yaml` expect the files below. Override any of them with
`dataset.params.<key>=<path>`.

| Dataset | Default path(s) under `raw/` |
| --- | --- |
| `squad` | `squad/dev-v2.0.json` |
| `hotpotqa` | `hotpotqa/hotpot_dev_distractor_v1.json` |
| `natural_questions` | `natural_questions/v1.0-simplified_nq-dev-all.jsonl.gz` |
| `ms_marco` | `ms_marco/collection.tsv`, `ms_marco/queries.dev.small.tsv`, `ms_marco/qrels.dev.small.tsv` |
| `academic_pdf` | `academic_pdf/**/*.pdf` |

Prepared datasets are written to run directories (`outputs/<experiment>/<run_id>/dataset/`),
not into this folder.
