---
id: "t000009"
name: "verify_single_retrieval_contradictions_with_additional_search"
description: "When a single sentence or claim from medrag_search appears to contradict a leading diagnosis that otherwise fits most clinical features, trigger a verification step before downgrading that diagnosis."
version: "1.0"
status: "active"
branch: "task_level"
task_category:
  - "differential_diagnosis"
applicable_conditions: "Differential diagnosis tasks where retrieval returns a seemingly discriminative feature that conflicts with a high–base-rate diagnosis or with the majority of findings in the case."
required_tools:
  - "medrag_search"
risk_level: "medium"
---

### Core Action
If a single claim retrieved via `medrag_search` appears to rule out or strongly argue against a leading diagnosis that otherwise matches most case features, do NOT immediately downgrade that diagnosis. Instead: (1) run at least two additional `medrag_search` queries—one targeting the diagnosis plus the disputed feature (e.g., "[diagnosis] with [feature]") and one targeting guideline or review summaries (e.g., "[diagnosis] typical features prevalence guideline"); (2) check whether multiple sources consistently support the exclusion claim and whether the feature truly contradicts the diagnosis or can occur in variants; (3) compare the overall feature fit and base-rate prevalence before adjusting the differential ranking. Only downgrade the leading diagnosis if corroborated evidence from multiple retrievals shows the feature reliably excludes it.
