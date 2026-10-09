# 🔧 Tool configuration

SkeMex registers **16 task tools + the automatic `describe_skill` helper**. Each benchmark exposes only its own tool subset. This guide describes the actual implementation and its service requirements; source code lives in [src/medmem_agent/tools](../src/medmem_agent/tools).

📋 [Tool directory](#-tool-directory) · 🔑 [Configuration](#-configure-services) · 📚 [Literature retrieval](#-literature-retrieval) · 🧪 [Local check](#-check-registration-without-api-calls)

## 🔑 Configure services

From the repository root:

```bash
cp .env.example .env
# Fill in credentials, endpoint roots, and paths locally.
set -a
source .env
set +a
```

- [global_config.json](global_config.json), `tools.shared`: shared credentials and model endpoint.
- `global_config.json`, `tools.<tool_type>`: each tool's model, timeouts, limits, and optional endpoint overrides.
- `datasets/<benchmark>/tools.json`: which tools the agent may call. Removing a tool here changes the benchmark setup.
- [llms/demo.json](llms/demo.json): the main agent's DeepSeek-V3.2 configuration. Reflection reuses this agent model by default.

**Priority:** global per-tool fields → mapped global shared fields → dataset-local fields → class defaults. Empty strings in the global blocks are omitted. In particular, the global PubMed/top-k=3 settings override the MedCorp/top-k=5 values in dataset files. API roots should include the provider's API prefix (usually `/v1`), without appending `/chat/completions`.

The shared endpoint must serve the configured models, including MiniMax synthesis/simulation and Qwen OCR; a provider-specific endpoint may serve only its own models. To use separate services, copy the complete global config to `configs/global_config.local.json` and add `api_key` / `api_base` fields to the relevant existing tool blocks. For example, these are **fields to merge**, not a complete replacement config:

```json
{
  "tools": {
    "ocr_chart_reader": {
      "api_key": "${OCR_API_KEY}",
      "api_base": "${OCR_BASE_URL}",
      "model": "qwen-vl-ocr-2025-11-20"
    },
    "medrag_search": {
      "api_key": "${SYNTHESIS_API_KEY}",
      "api_base": "${SYNTHESIS_BASE_URL}",
      "llm_name": "OpenAI/MiniMax-M2.7-highspeed"
    }
  }
}
```

Add those four variables to your local `.env`, reload it, and pass `--global_config configs/global_config.local.json` to the experiment script. Simulation tools also accept their own `api_key`, `api_base`, and `model` fields. Reflection needs all three explicitly if you choose a separate model. Credentials and `*.local.json` files are ignored by Git.

Provider model aliases must resolve to the intended model version. The configuration's `deepseek-v3.2` and `MiniMax-M2.7-highspeed` strings are deployment aliases, not a promise that every official endpoint currently accepts those names. See [DeepSeek API documentation](https://api-docs.deepseek.com/) and [MiniMax compatible API documentation](https://platform.minimax.io/docs/api-reference/text-openai-api).

## 📋 Tool directory

`tools.<name>` below refers to the corresponding block in `global_config.json`.

| Tool | Configuration / resource | Official resource or local implementation |
|---|---|---|
| `tavily_search` | `TAVILY_API_KEY`; `search_depth`, `topic`, `max_results`, `timeout_seconds`. Default: basic search, 3 results. | [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search) |
| `medrag_search` | Shared model endpoint for synthesis; `default_corpus`, `llm_name`, `top_k`. `NCBI_API_KEY` and `SEMANTIC_SCHOLAR_API_KEY` for search services. See routing below. | [NCBI E-utilities](https://eutilities.github.io/site/API_Key/usageandkey/), [Semantic Scholar API](https://api.semanticscholar.org/api-docs/), [MedRAG](https://github.com/gzxiong/MedRAG) |
| `drug_info_lookup` | `DRUGBANK_PATH` points to a local pandas `.pkl` table; falls back to RxNorm/openFDA when no local match is available. | [DrugBank downloads](https://go.drugbank.com/releases/latest), [RxNorm API](https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html), [openFDA labels](https://open.fda.gov/apis/drug/label/) |
| `drug_interaction_check` | Local DrugBank `.pkl` or `.xml`; set `drugbank_path`. Local interaction data is needed for reliable execution; see the fallback notes below. | [DrugBank XML schema](https://docs.drugbank.com/xml/#drug-interactions) |
| `umls_concept_lookup` | `UMLS_API_KEY`; `max_results`, `timeout_seconds`. Requires a UMLS account for full terminology lookup. | [UMLS authentication](https://documentation.uts.nlm.nih.gov/rest/authentication.html), [MedlinePlus Connect fallback](https://medlineplus.gov/medlineplus-connect/web-service/) |
| `python_calculator` | No key or service. Whitelisted arithmetic expressions, not an arbitrary Python interpreter. | [medical_calculators.py](../src/medmem_agent/tools/medical_calculators.py) |
| `unit_converter` | No key or service. Supply `value`, `from_unit`, `to_unit`; some conversions also require `substance`. | [medical_calculators.py](../src/medmem_agent/tools/medical_calculators.py) |
| `clinical_score_calculator` | No key or service. Supply `score_name` and the required numeric inputs; the tool schema lists supported scores and fields. | [medical_calculators.py](../src/medmem_agent/tools/medical_calculators.py) |
| `analyze_medical_image` | Hulu-Med-32B at `VISION_BASE_URL` / `VISION_API_KEY`; `model`, generation limits and timeout in its tool block. | [Hulu-Med model](https://huggingface.co/ZJU-AI4H/Hulu-Med-32B), [official code](https://github.com/ZJUI-AI4H/Hulu-Med) |
| `ocr_chart_reader` | `qwen-vl-ocr-2025-11-20` through the shared endpoint, or its own `api_base` / `api_key`. | [Qwen OCR setup and compatible API](https://www.alibabacloud.com/help/en/model-studio/qwen-vl-ocr) |
| `reflection` | Reuses the agent's main model when `model` is empty. A separate model requires explicit `model`, `api_key`, `api_base`. | [reasoning_tools.py](../src/medmem_agent/tools/reasoning_tools.py) |
| `verifier` | Shared endpoint; `model=deepseek-v3.2`. Override model/credentials in its block if needed. | [DeepSeek API](https://api-docs.deepseek.com/), [reasoning_tools.py](../src/medmem_agent/tools/reasoning_tools.py) |
| `agentclinic_patient_reply` | Shared endpoint; `model=MiniMax-M2.7-highspeed`. Patient context is supplied by the benchmark loader. | [AgentClinic](https://github.com/samuelschmidgall/agentclinic), [MiniMax API](https://platform.minimax.io/docs/api-reference/text-openai-api) |
| `agentclinic_request_exam` | Shared simulator endpoint/model; exams use the current case's runtime context. | [AgentClinic](https://github.com/samuelschmidgall/agentclinic) |
| `agentclinic_request_image` | `AGENTCLINIC_IMAGE_ROOT`; retrieves the current case's image reference. No standalone model call. | [AgentClinic](https://github.com/samuelschmidgall/agentclinic) |
| `mediq_patient_reply` | Shared simulator endpoint/model; case context is supplied by the MediQ loader. | [MediQ](https://github.com/stellalisy/MediQ) |
| `describe_skill` | Automatically registered; no credentials. Returns a tool's parameter schema and usage example. | [base.py](../src/medmem_agent/tools/base.py) |

Tool availability is controlled by the benchmark JSON; not all 17 tools are active for every dataset.

## 📚 Literature retrieval

The default literature-retrieval route is:

```text
corpus = PubMed
    → NCBI PubMed search/abstracts + Semantic Scholar search
    → merge and deduplicate by title
    → configured synthesis LLM
```

Each source returns up to `top_k` results, so the merged result can contain more than three documents. `NCBI_API_KEY` is read directly from the environment; it is optional for public requests under NCBI's limits. `SEMANTIC_SCHOLAR_API_KEY` is passed through the global shared block; requests without it use the public endpoint and its restrictions. See [NCBI key guidance](https://eutilities.github.io/site/API_Key/usageandkey/) and [Semantic Scholar API](https://api.semanticscholar.org/api-docs/).

**The default PubMed path does not require a local MedRAG corpus or MedCPT installation.** If you explicitly select another corpus, the code attempts local [MedRAG](https://github.com/gzxiong/MedRAG): install its dependencies separately, make its Python module importable, prepare the selected corpus/index, and set `MEDRAG_CORPUS_DIR`. Parameters include `retriever=MedCPT`, `HNSW=true` and `corpus_cache=true`. If local initialization fails, the tool falls back to online PubMed/Semantic Scholar. Record the actual source used in experiment outputs.

## 🗃️ Knowledge resources

**DrugBank:** acquire the data from [DrugBank](https://go.drugbank.com/releases/latest) under its own access terms. The default `DRUGBANK_PATH=drugbank/drugbank.pkl` supports both drug tools. Lookup expects a pandas table with a `synonyms` list per row and the available drug-information columns. Interaction checking additionally reads `drugbank_id`, `name`, `synonyms` and `drug_interactions`; consult [medical_knowledge.py](../src/medmem_agent/tools/medical_knowledge.py) for the loader schema. Only load a pickle you created or trust.

The interaction tool also accepts XML and writes a sibling pickle cache. The drug-information lookup does **not** parse XML directly: if you configure XML only for interactions, set that tool's `drugbank_path` separately. The retained RxNav interaction fallback targets a retired API; NLM [documents its discontinuation](https://lhncbc-portal.lhcaws-prod-pub.nlm.nih.gov/RxNav/information/FAQs.html). Do not treat an empty fallback result as a completed interaction lookup.

**UMLS:** the implementation uses ticket-based authentication. Current NLM documentation also describes direct API-key authentication; verify compatibility with your account. Missing credentials or unsuccessful UMLS lookup can fall back to MedlinePlus Connect, which is not an equivalent UMLS terminology result. Inspect the returned source field.

## 🖼️ Images and case simulation

Serve Hulu-Med using its official model/code instructions behind the vision-compatible endpoint expected by the tool; set both `VISION_*` variables. The analysis model alias must match your server. OCR uses the shared endpoint unless separately configured. Images must exist locally or be reachable by the configured service; see the [data guide](../data/README.md) for image roots.

AgentClinic/MediQ tools are tied to a case. The experiment runner calls `set_runtime_context` for each sample; do not call a simulator without the case fields prepared by the dataset loader. Reflection is automatically bound to the main agent model. A local heuristic fallback, missing-key response, or failed remote call is not evidence that the intended model service ran.

## 🧪 Check registration without API calls

After editable installation, run from the repository root. This lists names and schemas, never credentials, and performs no model or search calls:

```bash
python - <<'PY'
from medmem_agent.config import GlobalConfig
from medmem_agent.tools.registry import ToolRegistry

cfg = GlobalConfig.from_file('configs/global_config.json')
registry = ToolRegistry.from_config('configs/datasets/AgentClinic_MM/tools.json', cfg.tools)
print('Registered:', ', '.join(sorted(registry.tools)))
print(registry.run('describe_skill', {'skill_name': 'medrag_search'}).payload)
print(registry.run('python_calculator', {'expression': '2 + 3 * 4'}).payload)
PY
```

This checks local wiring only. Live endpoint availability, model aliases, dataset images and licensed corpora must be verified in your deployment. The repository does not download tool resources automatically.
