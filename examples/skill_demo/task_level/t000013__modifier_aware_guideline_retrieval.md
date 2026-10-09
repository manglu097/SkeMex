---
id: "t000013"
name: "Modifier_Aware_Guideline_Retrieval"
description: "When a clinical vignette includes a planned intervention or medication that could modify guideline recommendations, the agent must run a targeted retrieval that explicitly includes that modifier before finalizing advice."
version: "1.0"
status: "active"
branch: "task_level"
task_category:
  - "preventive_vaccination_and_travel_health_counseling"
applicable_conditions: "Clinical counseling or prevention tasks where a baseline rule exists (e.g., avoid screening asymptomatic individuals) but the vignette also introduces a modifier such as planned chronic medication use, upcoming therapy, comorbidity, or procedure that could change guideline recommendations."
required_tools:
  - "medrag_search"
risk_level: "low"
---

### Core Action
If an initial `medrag_search` retrieves a general rule (e.g., 'do not screen asymptomatic patients') and the vignette includes a management‑relevant modifier such as planned long‑term medication, immunosuppression, travel exposure, or another upcoming intervention, do NOT finalize the recommendation yet. Instead: (1) identify the modifier that could alter the guideline pathway; (2) run at least one additional `medrag_search` query that explicitly includes that modifier together with the condition or screening topic (e.g., "[condition] testing before long‑term [drug/class]"); (3) compare the modifier‑specific guidance with the baseline rule; (4) only then synthesize the final recommendation based on the most specific applicable guidance.
