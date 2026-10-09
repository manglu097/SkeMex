---
id: "g000009"
name: "Honor Explicit Output-Format Constraints"
description: "When the system or developer instruction specifies a strict output format (e.g., 'return strict JSON only'), the agent must treat that requirement as binding for the final response."
version: "1.0"
status: "active"
branch: "general"
task_category:
  - "non_clinical_social_interaction"
applicable_conditions: "Any task where system or developer instructions explicitly constrain the output format, structure, or serialization (e.g., strict JSON, schema-bound output, no prose, or machine-parsable responses)."
required_tools:
risk_level: "low"
---

### Core Action
When an instruction explicitly requires a specific output format (e.g., 'strict JSON only' or a provided schema), treat it as a hard constraint: produce the final message exactly in that format and nothing else. Do not include explanations, code fences, comments, or surrounding prose. Ensure the structure matches the required schema and that the output is valid and directly machine‑parsable.
