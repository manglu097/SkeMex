---
id: "a000004"
name: "avoid_redundant_tavily_query_repetition"
description: "When a tavily_search call has already returned results and the agent is about to issue the exact same query again."
version: "1.0"
status: "active"
branch: "action_level"
task_category:
  - "differential_diagnosis"
applicable_conditions: "Any workflow where web evidence is gathered using tavily_search and the previous call already produced results but the agent has not yet extracted or synthesized them."
required_tools:
  - "tavily_search"
risk_level: "low"
tool_name: "tavily_search"
---

### Core Action
If a `tavily_search` call has already returned results, do NOT repeat the identical query string. Instead: (1) first extract and synthesize the information from the existing results; or (2) if key aspects are still missing, reformulate the `query` to target the gap (e.g., add discriminating features, synonyms, or a comparison). Only issue a new search if the query changes. Example reformulated call: `{ "query": "causes of hemolytic anemia with jaundice differential diagnosis", "search_depth": "advanced" }` instead of repeating the earlier query unchanged.
