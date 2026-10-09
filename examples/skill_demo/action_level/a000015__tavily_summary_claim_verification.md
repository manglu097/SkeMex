---
id: "a000015"
name: "tavily_summary_claim_verification"
description: "When `tavily_search` returns an `answer` summary containing specific quantitative figures or policy claims."
version: "1.0"
status: "active"
branch: "action_level"
task_category:
  - "guideline_update_and_formula_adjustment"
applicable_conditions: "Any retrieval step where the search tool provides a synthesized answer field summarizing results and that summary includes concrete numbers, counts, dates, or policy assertions that could be propagated into the final response."
required_tools:
  - "tavily_search"
risk_level: "medium"
tool_name: "tavily_search"
---

### Core Action
When `tavily_search` returns a response containing an `answer` summary with specific numbers or policy claims, do NOT directly rely on that summary. Instead, run at least one verification search that targets the same claim and surfaces primary or corroborating sources. Reformulate the query to include the key claim elements (topic + number/policy + context) and use deeper retrieval. Example: `{"query": "how many states have universal newborn hearing screening policy United States", "search_depth": "advanced"}`. Only propagate the quantitative or policy claim after confirming it appears consistently in the underlying search results.
