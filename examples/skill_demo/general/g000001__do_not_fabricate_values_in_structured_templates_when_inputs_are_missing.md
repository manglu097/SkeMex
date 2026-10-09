---
id: "g000001"
name: "Do_Not_Fabricate_Values_In_Structured_Templates_When_Inputs_Are_Missing"
description: "When asked to generate a structured document or template but the prompt does not include concrete values for required fields."
version: "1.1"
status: "active"
branch: "general"
task_category:
  - "clinical_documentation_and_protocol_adherence"
applicable_conditions: "Tasks that request templates, forms, progress notes, reports, or structured documentation where typical fields (e.g., measurements, identifiers, timestamps, results, or status values) are expected but the user has not provided actual data for them."
required_tools:
risk_level: "medium"
---

### Core Action
If a prompt asks for a documentation template or structured note but specific field values are not provided, do NOT invent or auto-populate realistic-looking data (including default "normal" findings such as normal exam statements). Instead: (1) preserve the full structure of the document; (2) replace every field lacking provided data with an explicit placeholder such as "[not provided]", "[enter value]", or "<placeholder>"; (3) only include concrete values if they were explicitly given in the prompt; and (4) ensure that no section mixes placeholders with pre-filled clinical findings (e.g., default normal exam text), since this would present fabricated information as real observations.
