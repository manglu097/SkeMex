---
id: "t000063"
name: "systematic_multi_drug_interaction_screening"
description: "When performing a medication safety or interaction review for a patient taking multiple drugs, including combination products."
version: "1.0"
status: "active"
branch: "task_level"
task_category:
  - "drug_interaction_and_medication_safety_review"
applicable_conditions: "Cases where two or more medications are reported, especially when some are combination products or OTC analgesics with multiple active ingredients, and the task involves assessing medication safety, compatibility, or interaction risk."
required_tools:
  - "drug_interaction_check"
risk_level: "high"
---

### Core Action
When reviewing medication safety with multiple drugs, do NOT check only a single pair. First normalize the full medication list by expanding any combination products into their individual active ingredients. Then systematically screen interactions across the complete set by calling `drug_interaction_check` for every clinically meaningful pair (or the full medication list if the tool supports multi-drug input). Only after all relevant combinations have been evaluated should safety guidance or drug recommendations be produced.
