---
id: "a000028"
name: "medrag_search_warning_with_snippets_handling"
description: "When `medrag_search` returns an availability or warning message but still includes retrieved snippets."
version: "1.0"
status: "active"
branch: "action_level"
task_category:
  - "treatment_planning"
applicable_conditions: "Applies whenever the `medrag_search` tool response contains a warning, partial failure notice, or availability message while still returning one or more evidence snippets or document excerpts."
required_tools:
  - "medrag_search"
risk_level: "low"
tool_name: "medrag_search"
---

### Core Action
When `medrag_search` returns a response that includes a warning (e.g., availability or partial retrieval notice) but still contains non-empty `snippets` or retrieved text, treat the call as usable rather than failed. First read and extract evidence from the returned snippets and determine whether they support the treatment decision. Do NOT immediately retry the same query or switch tools unless the `snippets` field is empty or clearly irrelevant. Example call: `{ "query": "first line treatment for community acquired pneumonia adult", "top_k": 5 }`. If the response includes a warning but provides snippet text, proceed by analyzing those snippets before issuing any additional retrieval calls.
