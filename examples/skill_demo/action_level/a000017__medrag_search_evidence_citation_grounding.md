---
id: "a000017"
name: "medrag_search_evidence_citation_grounding"
description: "When summarizing or citing evidence from `medrag_search` results that only provide short snippets or summaries, the agent risks fabricating document identifiers or precise claims not present in the tool output."
version: "1.3"
status: "active"
branch: "action_level"
task_category:
  - "clinical_documentation_and_protocol_adherence"
applicable_conditions: "Applies whenever `medrag_search` returns summarized passages, snippets, or aggregated text without explicit document numbering, citation labels, or detailed evidence statements."
required_tools:
  - "medrag_search"
risk_level: "medium"
tool_name: "medrag_search"
---

### Core Action
When answering after a `medrag_search` call, treat the returned snippets as the only permitted evidence source. First read the snippet `text` fields and extract only statements that are explicitly present; quote or closely paraphrase those lines and attribute generically (e.g., "the retrieved summary states..."). Before including any quantitative value, percentage, dosage, outcome rate, or trial statistic, verify that the exact detail appears verbatim in the snippet text. If the snippets are high‑level summaries, appear truncated, or lack the specific treatment detail needed (e.g., efficacy rates, dosing, guideline thresholds), do NOT fill the gap using prior knowledge. Instead, run a follow‑up `medrag_search` with a more targeted query aimed at retrieving that exact detail (for example: {"query": "dupilumab randomized trial atopic dermatitis EASI-90 response rate", "top_k": 5}). Repeat retrieval until either (a) a snippet explicitly contains the needed claim or (b) multiple targeted searches still fail to surface it; in the latter case, explicitly state that the retrieved evidence does not provide that information and avoid inserting unsupported numbers or claims.
