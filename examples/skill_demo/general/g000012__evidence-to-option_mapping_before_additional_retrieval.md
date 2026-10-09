---
id: "g000012"
name: "Evidence-to-Option Mapping Before Additional Retrieval"
description: "When a prior step or tool output yields concrete discriminative features that already eliminate most candidate options."
version: "1.1"
status: "active"
branch: "general"
task_category:
  - "differential_diagnosis"
applicable_conditions: "Multiple-choice or candidate-selection tasks where earlier reasoning or tool outputs provide specific observable features, attributes, or constraints that clearly rule out several options."
required_tools:
risk_level: "low"
---

### Core Action
When a diagnostic tool output (e.g., `analyze_medical_image`) already returns concrete anatomical, mechanical, or procedural features relevant to the question, first perform an explicit evidence-to-option comparison before initiating any additional retrieval. Map each reported feature from the tool output against the answer choices and eliminate options that contradict those findings. If exactly one option remains consistent with all observed evidence, select it immediately and do NOT initiate `medrag_search` or other literature-retrieval loops. Only proceed to `medrag_search` if two or more options remain plausibly consistent with the tool-derived findings or if the tool output lacks discriminating detail.
