---
id: "skill_meta_003"
name: "Context Trim"
description: "Apply when the rendered prompt length exceeds 70% of the configured token budget."
version: "1.0"
status: "active"
branch: "meta_memory"
applicable_conditions: "When the rendered prompt token count exceeds 70% of the configured budget."
required_tools: []
risk_level: "medium"
created_at: "2026-04-07T00:00:00Z"
updated_at: "2026-04-07T00:00:00Z"
---

### Core Action
- Truncate observations in early steps to a short preview (keep first 200 chars + ellipsis).
- Preserve the most recent 3 steps in full.
- Keep all pinned findings intact — never truncate the `[Confirmed Findings]` block.

*Not applicable when*: the task is short and the prompt remains well below the token budget.
*Failure mode*: truncating numerical lab values or imaging findings may hurt reasoning; pin those findings before trimming.
