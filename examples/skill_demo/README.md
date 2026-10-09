# 🧩 Evolved skill examples

**15 hand-picked examples · 5 per branch**

These examples show what SkeMex's evolved skills look like: a trigger, structured metadata, and a reusable **Core Action**. They are representative illustrations, not a complete checkpoint for reproducing benchmark scores.

The skill text and descriptive metadata are preserved. Source trajectory identifiers and run timestamps have been removed. No utility index, embedding cache, interaction trace, or meta-memory is included. The experiment scripts do not load this directory automatically.

## 🧠 General — reusable reasoning principles

| Example | What it illustrates |
|---|---|
| [g000001 · Missing values in structured documents](general/g000001__do_not_fabricate_values_in_structured_templates_when_inputs_are_missing.md) | Keep placeholders instead of inventing observations. |
| [g000003 · Evidence before changing hypotheses](general/g000003__evidence_threshold_before_hypothesis_switch.md) | Require supporting evidence before switching a working hypothesis. |
| [g000005 · Plans with missing constraints](general/g000005__avoid_single_plan_when_safety_constraints_unknown.md) | Use conditional options when key constraints are unknown. |
| [g000009 · Explicit output formats](general/g000009__honor_explicit_output-format_constraints.md) | Respect a required output format or schema. |
| [g000012 · Evidence-to-option mapping](general/g000012__evidence-to-option_mapping_before_additional_retrieval.md) | Compare existing evidence with candidates before more retrieval. |

## 🩺 Task-level — clinical task procedures

| Example | What it illustrates |
|---|---|
| [t000009 · Differential diagnosis](task_level/t000009__verify_single_retrieval_contradictions_with_additional_search.md) | Verify a retrieved contradiction before changing the diagnosis ranking. |
| [t000037 · Guideline reconciliation](task_level/t000037__guideline_source_reconciliation_for_clinical_recommendation_queries.md) | Keep differing organizations’ recommendations clearly attributed. |
| [t000063 · Medication safety](task_level/t000063__systematic_multi_drug_interaction_screening.md) | Expand combination products and screen the medication list systematically. |
| [t000030 · Medical image interpretation](task_level/t000030__require_image_before_imaging_interpretation.md) | Distinguish missing-image tasks from text-only questions about expected findings. |
| [t000013 · Preventive counseling](task_level/t000013__modifier_aware_guideline_retrieval.md) | Include patient-context modifiers in guideline retrieval. |

## 🛠️ Action-level — tool-specific practices

| Example | What it illustrates |
|---|---|
| [a000011 · Validate a drug lookup result](action_level/a000011__drug_info_lookup_result_identity_validation.md) | Check that the returned drug identity matches the requested medication. |
| [a000015 · Verify a Tavily summary](action_level/a000015__tavily_summary_claim_verification.md) | Check quantitative or policy claims against search results. |
| [a000017 · Ground MedRAG citations](action_level/a000017__medrag_search_evidence_citation_grounding.md) | Use retrieved evidence rather than inventing citation details or statistics. |
| [a000028 · Handle partial MedRAG results](action_level/a000028__medrag_search_warning_with_snippets_handling.md) | Inspect usable snippets before retrying a warning-bearing response. |
| [a000004 · Avoid repeated Tavily queries](action_level/a000004__avoid_redundant_tavily_query_repetition.md) | Use existing results or reformulate the query to address an evidence gap. |

## 📖 Reading a skill

Each Markdown file contains an ID, name, description, branch, task category, applicability conditions, required tools, risk level, and the learned action. Tool-specific skills also identify their tool. Version and status are metadata, not validation scores.

These are illustrative learned instructions. For the callable tool interfaces, see the [tool configuration guide](../../configs/README.md).

[← SkeMex](../../README.md)
