---
id: "t000030"
name: "require_image_before_imaging_interpretation"
description: "When a task asks for interpretation of a medical image but no image file or accessible image data is provided."
version: "1.1"
status: "active"
branch: "task_level"
task_category:
  - "medical_imaging_interpretation"
applicable_conditions: "Tasks in medical imaging interpretation (e.g., radiology, pathology slides, dermatology photos, ophthalmology images) where the prompt requests diagnostic or descriptive interpretation but the actual image input is missing, inaccessible, or not attached."
required_tools:
  - "analyze_medical_image"
risk_level: "high"
---

### Core Action
When a request asks for interpretation of a specific medical image that is expected to be visually examined (e.g., the prompt references an attached CT/MRI/X‑ray, a provided image file, or asks what *this image* shows) but no image is actually available, do NOT infer findings from general knowledge. Instead: (1) request the image or confirm that an image file is available; (2) wait until the image is provided; (3) call `analyze_medical_image` with the image input to extract visual findings; (4) base any diagnostic reasoning only on those returned results. However, if the question is a knowledge-based clinical inference about what imaging would most likely show (e.g., exam-style vignettes asking for the expected CT/MRI finding, lesion location, or imaging appearance based solely on symptoms and clinical context) and no specific image is referenced as provided, proceed with standard medical reasoning using the textual information and do NOT request an image.
