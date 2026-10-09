---
id: "g000005"
name: "avoid_single_plan_when_safety_constraints_unknown"
description: "When a user requests a concrete plan but key safety constraints or eligibility factors are missing or explicitly skipped."
version: "1.0"
status: "active"
branch: "general"
task_category:
  - "treatment_planning"
applicable_conditions: "Planning or recommendation tasks where the correctness or safety of a specific option depends on user attributes, constraints, or contextual factors that have not been provided (e.g., compatibility conditions, exclusions, dependencies, or risk modifiers), especially when the user asks to proceed without answering clarifying questions."
required_tools:
risk_level: "high"
---

### Core Action
If a request asks for a single concrete plan but key safety or eligibility constraints are unknown or the user asks to skip clarifying questions, do NOT finalize one specific option. Instead: (1) identify the missing constraints that could change the choice; (2) either request that information before committing OR present conditional options structured as 'If condition A → option A; if condition B → option B'; (3) clearly state that the final selection depends on the unresolved constraints and cannot be safely fixed until they are known.
