# ⚙️ Configuration and services

All commands run from the source checkout root after `pip install -e .`.
JSON is parsed before environment substitution, including nested values and lists.
`.env` is loaded explicitly by the shell; the library does not load it automatically.

## Service separation

- `configs/llms/demo.json`: DeepSeek-V3.2 agent configuration.
- `configs/evolution/default.json`: shared evolution settings; pre-screening, category classification, skill writing, and governance use DeepSeek-V3.2. API credentials come from `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
- `configs/evolution/offline.json` / `online.json`: mode, input roots, experiment directories, and schedules. Defaults are merged with the selected mode file.
- `configs/eval/default.json`: Gemini-3-Flash semantic/rubric judge, with independent `JUDGE_*` credentials. `gemini-3-flash-preview` is a portable model alias; match the alias and decoding behavior to the service you use.
- `EMBEDDING_*` overrides the evolution service for embeddings. The main pipeline requests `text-embedding-3-large`.
- `configs/global_config.json`: tool service settings. Global tool values override dataset-local values. This matters for MedRAG's PubMed corpus and top-k=3.
- `configs/datasets/*/tools.json`: allowed tool set for each benchmark. Image tools are available only for appropriate configurations.

API base URLs must be OpenAI-compatible roots, typically ending in `/v1`.
Set service-specific keys and URLs together. A custom endpoint may require a different model alias or `extra_body`; those settings are provider-specific and must be recorded with each run.
Provider-specific extensions such as `enable_thinking` are intentionally not included; add them only when the selected backend documents and supports them.

## Tool setup

See [the tool configuration README](../configs/README.md) for all 17 tools, exact configuration fields, official resources, endpoint overrides, and a local registration check.

The supplied `medrag_search` configuration uses PubMed E-utilities and Semantic Scholar, merged and summarized by an LLM. Local MedRAG installation is needed only when selecting another corpus. The shared endpoint must serve the configured synthesis/simulation/OCR models, or those tools must be assigned their own endpoints.

## Local files

`SKEMEX_DATA_ROOT` locates the original image directories for MMMU/MMMU-Pro.
`AGENTCLINIC_IMAGE_ROOT` supplies AgentClinic's image store.
Other loaders resolve images from their dataset directory: preserve the processed directory layout and inspect image existence before running.
Do not upload local configs, API responses, embedding caches, or skill usage traces.

For reproducibility, save the exact non-secret configuration, model version, prompt version, data checksums, and repository revision with each run.
