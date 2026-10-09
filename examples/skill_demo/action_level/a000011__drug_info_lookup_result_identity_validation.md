---
id: "a000011"
name: "drug_info_lookup_result_identity_validation"
description: "When a `drug_info_lookup` response contains information for a different medication than the one requested."
version: "1.0"
status: "active"
branch: "action_level"
task_category:
  - "perioperative_patient_education_and_counseling"
applicable_conditions: "Any workflow where medication knowledge is retrieved via `drug_info_lookup` and the returned text references a drug name, brand, or indications inconsistent with the queried medication."
required_tools:
  - "drug_info_lookup"
risk_level: "high"
tool_name: "drug_info_lookup"
---

### Core Action
If `drug_info_lookup` is called and the returned content references a different drug than the queried medication (e.g., query `{"drug_name": "mycophenolate"}` but the response describes cyclosporine), treat the result as invalid and do NOT use it for reasoning. Immediately retry the lookup by correcting or expanding the parameter (e.g., `{"drug_name": "mycophenolate mofetil"}` or a known brand such as `{"drug_name": "CellCept"}`). Only proceed with reasoning after confirming the returned entry clearly matches the intended medication name or synonyms.
