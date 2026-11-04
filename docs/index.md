# IPS-R documentation

| Page | Contents |
| --- | --- |
| [../AGENTS.md](../AGENTS.md) | Binding engineering rules |
| [architecture.md](architecture.md) | Package structure, component interfaces, dataset layer, knowledge pipeline, extension guide |
| [configuration.md](configuration.md) | Hydra config groups (datasets, chunkers, preparation, knowledge pipeline), overrides, environment variables |
| [reproducibility.md](reproducibility.md) | Seeds, run directories, metadata, content hashes, deterministic knowledge artifacts |
| [development.md](development.md) | Tooling, quality gates, CI, Docker |

## Status

| Stage | Scope | Status |
| --- | --- | --- |
| Sprint 1 | Infrastructure and interfaces | Done |
| Milestone 1 | Dataset layer: ingestion, validation, canonical models, PDF parsing, chunking, statistics | Done |
| Milestone 2 | Knowledge processing pipeline: normalisation, document structure, knowledge objects, chunk processing, enrichment, embedding/index interfaces, serialization, hashing, statistics | Done |
| Milestone 3+ | Retrieval and later research components | Not started |
