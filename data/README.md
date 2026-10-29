# Data

Git ignores the contents of this folder, apart from the directory skeleton.

| Folder | Contents |
| --- | --- |
| `raw/` | Original, immutable source data |
| `external/` | Third-party resources (e.g. pre-built indices, word lists) |
| `interim/` | Intermediate transformations |
| `processed/` | Final, model-ready datasets |

To relocate it, for example to a scratch disk, set `IPS_R_DATA_DIR`.
