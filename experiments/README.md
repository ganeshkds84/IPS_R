# Experiments

Use this folder for experiment **plans, notes and analysis**. Use one sub-folder per study:

```
experiments/
└── E001-short-name/
    ├── README.md      hypothesis, setup, exact commands, conclusion
    └── analysis.md    tables and figures pulled from outputs/
```

Executable experiment **definitions** are Hydra presets in
[`configs/experiment/`](../configs/experiment/). Run one with:

```bash
uv run ips-r run experiment=<preset>
```

Run artefacts (config snapshot, metadata, logs, results) are written to
`outputs/<experiment.name>/<run_id>/`, which git ignores. Reference run IDs from
the notes here.
