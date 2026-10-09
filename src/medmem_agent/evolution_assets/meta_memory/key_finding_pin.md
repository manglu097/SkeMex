---
id: "skill_meta_002"
name: "Key Finding Pin"
description: "Apply once at step 5 to extract and pin confirmed high-value medical findings for persistent context injection in later steps."
version: "1.0"
status: "active"
branch: "meta_memory"
applicable_conditions: "When current step index equals 5 and the task has not yet completed."
required_tools: []
risk_level: "low"
created_at: "2026-04-07T00:00:00Z"
updated_at: "2026-04-07T00:00:00Z"
---

### Core Action
- Fires once when current step index equals 5.
- Scan all early-step observations and extract confirmed high-value medical facts (labs, vitals, imaging, allergy history, guideline lookups).
- Format as a `[Confirmed Findings]` block and pin for all subsequent steps.
- Keep the block to 3–5 bullets; discard speculative or pending results.

*Not applicable when*: the task completes within 5 steps.
*Failure mode*: over-extraction bloats the pinned block and wastes token budget — include only findings that are explicitly confirmed in observations.
