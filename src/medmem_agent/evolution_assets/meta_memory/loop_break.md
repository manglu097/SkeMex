---
id: "skill_meta_001"
name: "Loop Break Injection"
description: "Apply when the last 3 consecutive steps share identical action and action_input, indicating the agent is stuck in a tool-call loop."
version: "1.0"
status: "active"
branch: "meta_memory"
applicable_conditions: "When steps[-3:] all share identical action and action_input."
required_tools: []
risk_level: "low"
created_at: "2026-04-07T00:00:00Z"
updated_at: "2026-04-07T00:00:00Z"
---

### Core Action
- Check `steps[-3:]`: if all three share identical `action` and `action_input`, fire this guard.
- Append a one-time system notice to the next step prompt (no extra model call required).
- Auto-clear the notice after the next step executes.

*Not applicable when*: repeated calls are intentional (e.g., polling async results or streaming tool outputs).
