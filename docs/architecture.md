# Architecture (Sprint 1)

Sprint 1 provides the *scaffolding* that the frozen research architecture plugs
into. It defines interfaces only: no retrieval, ranking, policy or generation
logic is implemented.

## Layers

```
cli.py ──► experiment/runner ──► pipeline/builder ──► registries ──► component implementations
   │              │                                                  (later sprints)
   ▼              ▼
config/loader   core/{log, seed, paths, env}
```

- **core**: dependency-free infrastructure (errors, logging, env, paths, seeds, registry, data types).
- **config**: the `AppConfig` schema and Hydra compose-API loader.
- **components**: `AbstractComponent`, the shared base class.
- **interfaces**: one package per component kind, each with an abstract base and a registry.
- **pipeline**: `PipelineComponents`, `AbstractPipeline` and the config-driven builder.
- **experiment**: run lifecycle (directories, config snapshot, metadata, execution).

## Component interfaces

| Interface | Package | Public method(s) | Hook(s) to implement | Registry |
| --- | --- | --- | --- | --- |
| `AbstractDataset` | `ips_r.data` | `queries()`, `documents()`, `references()` | `_load_queries`, `_load_documents`, `_load_references` | `DATASETS` |
| `AbstractRetriever` | `ips_r.retrieval` | `index()`, `retrieve()`, `retrieve_batch()` | `_index`, `_retrieve` | `RETRIEVERS` |
| `AbstractCandidateGenerator` | `ips_r.candidates` | `generate()` | `_generate` | `CANDIDATE_GENERATORS` |
| `AbstractDecisionPolicy` | `ips_r.policy` | `decide()` | `_decide` | `DECISION_POLICIES` |
| `AbstractReranker` | `ips_r.reranking` | `rerank()` | `_rerank` | `RERANKERS` |
| `AbstractGenerator` | `ips_r.generation` | `generate()`, `generate_batch()` | `_generate` | `GENERATORS` |
| `AbstractEvaluator` | `ips_r.evaluation` | `evaluate()` | `_evaluate` | `EVALUATORS` |
| `AbstractPipeline` | `ips_r.pipeline` | `run()` | `_run` | `PIPELINES` |

All of them follow the **template method** pattern. The public method:

1. validates arguments (e.g. `top_k > 0`, aligned predictions/references),
2. runs `setup()` lazily the first time,
3. logs start/end with timing,
4. calls the protected hook,
5. checks the output contract (e.g. at most `top_k` results, reranker output ⊆ input),
6. wraps unexpected exceptions in `ComponentError`, chaining the original as `__cause__`.

Implementations therefore contain only research logic.

### Data contracts

`ips_r.core.types` defines immutable containers: `Query`, `Document`,
`ScoredDocument`, `Candidate`, `Decision`, `Generation`, `Reference` and
`EvaluationResult`. They are **provisional**. Each carries a read-only
`metadata` mapping, so later sprints can attach extra information without
changing the interfaces.

### Lifecycle

```python
with MyRetriever({"k1": 1.2}) as retriever:   # setup() ... teardown()
    retriever.index(documents)
    hits = retriever.retrieve(query, top_k=10)
```

Override `_setup` / `_teardown` to load or release models and indices.
`AbstractPipeline` cascades both to all of its components.

## Adding an implementation (later sprints)

1. Implement the interface and register it:

   ```python
   # src/ips_r/retrieval/my_retriever.py
   from ips_r.retrieval import RETRIEVERS, AbstractRetriever

   @RETRIEVERS.register("my_retriever")
   class MyRetriever(AbstractRetriever):
       def _index(self, documents): ...
       def _retrieve(self, query, top_k): ...
   ```

2. Import the module from the package's `__init__.py` so that registration runs.
3. Add `configs/retriever/my_retriever.yaml`:

   ```yaml
   name: my_retriever
   params:
     k1: 1.2
   ```

4. Select it with `ips-r run retriever=my_retriever`.
5. Add tests under `tests/unit/`.
