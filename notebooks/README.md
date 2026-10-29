# Notebooks

Exploratory analysis only. Logic that matters belongs in `src/ips_r/` with tests.

Conventions:

- Name notebooks `NN-author-topic.ipynb`, e.g. `01-gk-data-exploration.ipynb`.
- Load configuration the same way the CLI does:

  ```python
  from ips_r.config import load_config
  from ips_r.core.seed import seed_everything

  cfg = load_config(["experiment=example"])
  seed_everything(cfg.seed.value)
  ```

- Clear outputs before committing large notebooks.

Jupyter is not a project dependency. Run it ad hoc with
`uv run --with jupyterlab jupyter lab`.
