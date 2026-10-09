---
id: "t000037"
name: "Guideline Source Reconciliation for Clinical Recommendation Queries"
description: "When answering a clinical guideline question using multiple search tools, conflicting recommendations from different guideline organizations may appear and must be explicitly reconciled rather than merged."
version: "1.0"
status: "active"
branch: "task_level"
task_category:
  - "guideline_update_and_formula_adjustment"
applicable_conditions: "Queries asking for screening ages, treatment thresholds, monitoring intervals, or other recommendations governed by professional guideline bodies where multiple organizations may publish differing guidance."
required_tools:
  - "medrag_search"
  - "tavily_search"
risk_level: "medium"
---

### Core Action
When a clinical guideline question requires external lookup, first call `medrag_search` to retrieve medical literature or guideline summaries. If the answer is incomplete or uncertain, call `tavily_search` to locate additional authoritative sources (e.g., USPSTF, ACS, AUA, specialty societies). After both searches, do NOT average or generalize recommendations. Instead: (1) identify the specific guideline organizations referenced in the retrieved sources; (2) extract each organization's stated recommendation (e.g., age ranges, risk groups, or screening intervals); (3) prioritize authoritative guideline bodies and present their thresholds separately if they differ; and (4) anchor the final answer explicitly to the named organizations (e.g., 'USPSTF recommends X…, ACS recommends Y…'). Only synthesize when sources truly agree.
