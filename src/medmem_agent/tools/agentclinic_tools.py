import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import DEFAULT_API_BASE, ToolParam, ToolResult, call_tool_llm

logger = logging.getLogger(__name__)


# ===========================================================================
# Shared helpers
# ===========================================================================


def _agentclinic_context(runtime_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(runtime_context, dict):
        return {}
    nested = runtime_context.get("agentclinic")
    if isinstance(nested, dict):
        return nested
    return runtime_context



def _get_context_value(runtime_context: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    context = _agentclinic_context(runtime_context)
    for key in keys:
        if key in context and context[key] not in (None, ""):
            return context[key]
    return default



def _load_case_from_payload(payload: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    case_data = payload.get("case_data")
    if case_data is None:
        case_data = _get_context_value(
            runtime_context,
            "agentclinic_case",
            "case_data",
            "current_case",
        )

    if not isinstance(case_data, dict):
        raise ValueError(
            "No AgentClinic case is bound to this tool call. "
            "Bind the current case through ToolRegistry.set_runtime_context({'agentclinic_case': case_dict}) "
            "or AgentLoop.run(..., tool_runtime_context={'agentclinic_case': case_dict})."
        )
    return case_data



def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_flatten_text(v) for v in value]
        return "; ".join([p for p in parts if p])
    if isinstance(value, dict):
        parts: List[str] = []
        for key, val in value.items():
            flat = _flatten_text(val)
            if flat:
                parts.append(f"{key.replace('_', ' ')}: {flat}")
        return "; ".join(parts)
    return str(value)



def _humanise_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip()



def _normalise_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()



def _tokenise(text: str) -> List[str]:
    normalised = _normalise_label(text)
    return [tok for tok in normalised.split() if tok]



def _score_label_match(query: str, label: str) -> int:
    qn = _normalise_label(query)
    ln = _normalise_label(label)
    if not qn or not ln:
        return 0
    if qn == ln:
        return 100
    if qn in ln or ln in qn:
        return 88

    q_tokens = set(_tokenise(qn))
    l_tokens = set(_tokenise(ln))
    if not q_tokens or not l_tokens:
        return 0

    overlap = len(q_tokens & l_tokens)
    if overlap == 0:
        return 0

    coverage = overlap / max(len(q_tokens), len(l_tokens))
    return min(84, int(100 * coverage))



def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return [chunk.strip() for chunk in chunks if chunk.strip()]



def _limit_sentences(text: str, max_sentences: int) -> str:
    if max_sentences <= 0:
        return text.strip()
    sentences = _split_sentences(text)
    if not sentences:
        return text.strip()
    return " ".join(sentences[:max_sentences]).strip()



# Keep a module-level alias so that ``mediq_tools.py`` (and any other code
# that imported ``_call_text_llm`` from this module) continues to work.
_call_text_llm = call_tool_llm


_AGENTCLINIC_PATIENT_BIAS_TEXT = {
    "none": "No patient-side bias should be applied.",

    "self_diagnosis": (
        "The patient may mention that they have researched their symptoms and "
        "formed a tentative personal explanation, which may influence how they "
        "respond to the doctor."
    ),

    "recency": (
        "The patient may refer to a recently heard story or diagnosis involving "
        "similar symptoms, which may subtly influence their concerns."
    ),

    "frequency": (
        "The patient may lean toward what they believe is the most common "
        "explanation for their symptoms based on what they have heard."
    ),

    "false_consensus": (
        "The patient may mention that family members or friends share a similar "
        "concern about their symptoms."
    ),

    "gender": (
        "The patient may sound slightly guarded or less trusting due to perceived "
        "gender-related differences, without changing factual information."
    ),

    "race": (
        "The patient may sound slightly guarded or less trusting due to perceived "
        "racial differences, without changing factual information."
    ),

    "sexual_orientation": (
        "The patient may sound slightly guarded or less trusting due to perceived "
        "differences in sexual orientation, without changing factual information."
    ),

    "cultural": (
        "The patient may sound slightly guarded or cautious due to cultural "
        "differences, without changing factual information."
    ),

    "education": (
        "The patient may occasionally second-guess explanations or ask for "
        "additional clarification, without changing factual information."
    ),

    "religion": (
        "The patient may sound slightly guarded or cautious due to perceived "
        "religious differences, without changing factual information."
    ),

    "socioeconomic": (
        "The patient may sound slightly guarded or cautious due to perceived "
        "socioeconomic differences, without changing factual information."
    ),
}

_AGENTCLINIC_DOCTOR_BIAS_TEXT = {
    "none": "No doctor-side bias note is applied.",
    "confirmation": "The doctor may be anchoring on an early hunch. The patient should answer naturally and not adapt facts to fit that hunch.",
    "status_quo": "The doctor may favour routine explanations. The patient should not change facts in response.",
    "recency": "The doctor may be influenced by recent cases. The patient should not change facts in response.",
    "frequency": "The doctor may favour common explanations. The patient should not change facts in response.",
    "false_consensus": "The doctor may carry group assumptions. The patient should not change facts in response.",
    "gender": "Doctor bias should not change patient facts.",
    "race": "Doctor bias should not change patient facts.",
    "sexual_orientation": "Doctor bias should not change patient facts.",
    "cultural": "Doctor bias should not change patient facts.",
    "education": "Doctor bias should not change patient facts.",
    "religion": "Doctor bias should not change patient facts.",
    "socioeconomic": "Doctor bias should not change patient facts.",
}


class AgentClinicRuntimeContextMixin:
    def set_runtime_context(self, context: Optional[Dict[str, Any]]) -> None:
        self._runtime_context = dict(context or {})


class AgentClinicCaseAdapter:
    """Normalise AgentClinic MedQA and NEJM records into one internal schema."""

    @staticmethod
    def normalise(case_data: Dict[str, Any]) -> Dict[str, Any]:
        if "OSCE_Examination" in case_data:
            osce = case_data["OSCE_Examination"]
            patient_info = osce.get("Patient_Actor", {})
            exams = osce.get("Physical_Examination_Findings", {})
            tests = osce.get("Test_Results", {})
            diagnosis = osce.get("Correct_Diagnosis", "")
            return {
                "dataset_variant": "agentclinic_medqa",
                "case_question": osce.get("Objective_for_Doctor", ""),
                "patient_info": patient_info,
                "patient_text": _flatten_text(patient_info),
                "physical_exams": exams,
                "physical_exam_text": _flatten_text(exams),
                "test_results": tests,
                "test_results_text": _flatten_text(tests),
                "gold_answer": diagnosis,
                "image_path": case_data.get("image_path") or osce.get("image_path") or "",
                "image_url": case_data.get("image_url") or osce.get("image_url") or "",
                "answer_options": [],
            }

        answers = case_data.get("answers", [])
        gold = ""
        for ans in answers:
            if isinstance(ans, dict) and ans.get("correct"):
                gold = str(ans.get("text", "")).strip()
                break

        patient_info_raw = case_data.get("patient_info", "")
        physical_exams_raw = case_data.get("physical_exams", "")
        return {
            "dataset_variant": "agentclinic_nejm",
            "case_question": case_data.get("question", ""),
            "patient_info": {"patient_info": patient_info_raw},
            "patient_text": _flatten_text(patient_info_raw),
            "physical_exams": {"physical_exams": physical_exams_raw},
            "physical_exam_text": _flatten_text(physical_exams_raw),
            "test_results": {},
            "test_results_text": "",
            "gold_answer": gold,
            "image_path": case_data.get("image_path", ""),
            "image_url": case_data.get("image_url", ""),
            "answer_options": answers,
        }

    @staticmethod
    def build_patient_prompt_context(normalised_case: Dict[str, Any]) -> str:
        if normalised_case["dataset_variant"] == "agentclinic_medqa":
            patient = normalised_case["patient_info"]
            symptoms = patient.get("Symptoms", {}) if isinstance(patient, dict) else {}
            lines = [
                f"Case objective for doctor: {normalised_case.get('case_question', '')}",
                f"Demographics: {_flatten_text(patient.get('Demographics', ''))}",
                f"History: {_flatten_text(patient.get('History', ''))}",
                f"Primary symptom: {_flatten_text(symptoms.get('Primary_Symptom', ''))}",
                f"Secondary symptoms: {_flatten_text(symptoms.get('Secondary_Symptoms', []))}",
                f"Past medical history: {_flatten_text(patient.get('Past_Medical_History', ''))}",
                f"Social history: {_flatten_text(patient.get('Social_History', ''))}",
                f"Review of systems: {_flatten_text(patient.get('Review_of_Systems', ''))}",
            ]
            return "\n".join([line for line in lines if line.split(":", 1)[-1].strip()])

        lines = [
            f"Case question: {normalised_case.get('case_question', '')}",
            f"Patient-visible information: {normalised_case.get('patient_text', '')}",
        ]
        return "\n".join([line for line in lines if line.split(":", 1)[-1].strip()])

    @staticmethod
    def _aliases_for_path(path_parts: List[str], section: str) -> List[str]:
        human_parts = [_humanise_key(part) for part in path_parts if _humanise_key(part)]
        aliases: List[str] = []

        if human_parts:
            aliases.append(" ".join(human_parts))
            aliases.append(human_parts[-1])
            for idx in range(len(human_parts)):
                aliases.append(" ".join(human_parts[idx:]))

        joined = _normalise_label(" ".join(human_parts))
        if "vital signs" in joined:
            aliases.extend(["vital signs", "vitals"])
        if "neurological examination" in joined:
            aliases.extend(["neurological examination", "neurological exam", "neuro exam"])
        if "blood tests" in joined:
            aliases.extend(["blood tests", "laboratory results", "lab results", "lab tests", "blood work", "labs"])
        if "electromyography" in joined:
            aliases.extend(["electromyography", "emg"])
        if "imaging" in joined:
            aliases.extend(["imaging", "radiology", "scan"])
        if "chest ct" in joined:
            aliases.extend(["chest ct", "ct chest", "ct scan", "ct"])
        if section == "physical_exam":
            aliases.extend(["physical examination", "physical exam"])
        if section == "test_result":
            aliases.extend(["test results", "tests", "investigations"])

        seen = set()
        unique: List[str] = []
        for alias in aliases:
            norm = _normalise_label(alias)
            if norm and norm not in seen:
                unique.append(alias)
                seen.add(norm)
        return unique

    @staticmethod
    def list_exam_items(normalised_case: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        seen = set()

        def _add_item(section: str, label: str, value: Any, aliases: Optional[List[str]] = None) -> None:
            text = _flatten_text(value)
            label_h = _humanise_key(label)
            if not label_h or not text:
                return
            key = (section, _normalise_label(label_h), text)
            if key in seen:
                return
            seen.add(key)
            items.append(
                {
                    "section": section,
                    "label": label_h,
                    "value": text,
                    "aliases": aliases or [],
                }
            )

        def _walk(obj: Any, path_parts: List[str], section: str) -> None:
            if isinstance(obj, dict):
                if path_parts:
                    _add_item(
                        section,
                        " ".join([_humanise_key(p) for p in path_parts]),
                        obj,
                        aliases=AgentClinicCaseAdapter._aliases_for_path(path_parts, section),
                    )
                for key, val in obj.items():
                    _walk(val, path_parts + [key], section)
            else:
                _add_item(
                    section,
                    " ".join([_humanise_key(p) for p in path_parts]),
                    obj,
                    aliases=AgentClinicCaseAdapter._aliases_for_path(path_parts, section),
                )

        physical_exams = normalised_case.get("physical_exams", {})
        test_results = normalised_case.get("test_results", {})

        if normalised_case["dataset_variant"] == "agentclinic_medqa":
            if physical_exams:
                _add_item(
                    "physical_exam",
                    "physical examination",
                    physical_exams,
                    aliases=["physical examination", "physical exam", "exam", "examination", "physical findings"],
                )
                _walk(physical_exams, [], "physical_exam")

            if test_results:
                _add_item(
                    "test_result",
                    "test results",
                    test_results,
                    aliases=["test results", "tests", "laboratory results", "lab results", "labs", "blood work", "investigations"],
                )
                _walk(test_results, [], "test_result")
        else:
            if physical_exams:
                _add_item(
                    "physical_exam",
                    "physical exams",
                    physical_exams,
                    aliases=[
                        "physical exams",
                        "physical examination",
                        "physical exam",
                        "dermoscopy",
                        "skin biopsy",
                        "biopsy",
                        "test results",
                        "exam",
                        "examination",
                    ],
                )

        return items


# ===========================================================================
# Tool 1: AgentClinic patient reply
# ===========================================================================


@dataclass
class AgentClinicPatientReplyTool(AgentClinicRuntimeContextMixin):
    name: str = "agentclinic_patient_reply"
    description: str = (
            "Call this tool when you need additional information from the patient. "
            "Use it to ask questions about symptoms, medical history,"
            "or other relevant clinical details."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 256
    timeout_seconds: int = 60
    max_sentences: int = 3
    _runtime_context: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_base:
            self.api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)

    def _build_prompts(
        self,
        *,
        normalised_case: Dict[str, Any],
        doctor_query: str,
        patient_bias: str,
        doctor_bias: str,
    ) -> Tuple[str, str, List[str]]:
        patient_context = AgentClinicCaseAdapter.build_patient_prompt_context(normalised_case)
        pb = (patient_bias or "none").strip().lower()
        db = (doctor_bias or "none").strip().lower()
        bias_notes: List[str] = []

        if pb in _AGENTCLINIC_PATIENT_BIAS_TEXT:
            bias_notes.append(_AGENTCLINIC_PATIENT_BIAS_TEXT[pb])
        elif pb:
            bias_notes.append(f"Unsupported patient bias '{patient_bias}' ignored.")

        if db in _AGENTCLINIC_DOCTOR_BIAS_TEXT:
            bias_notes.append(_AGENTCLINIC_DOCTOR_BIAS_TEXT[db])
        elif db:
            bias_notes.append(f"Unsupported doctor bias '{doctor_bias}' ignored.")

        system_prompt = (
            "You are role-playing the patient in an AgentClinic clinical benchmark. "
            "Answer the doctor's question naturally in first person, as the patient only. "
            "Use ONLY the patient-visible information provided. "
            "Do NOT invent new symptoms, timing, exposures, medications, diagnoses, exam findings, lab values, imaging results, or treatments. "
            "Do NOT reveal hidden benchmark labels or the correct diagnosis. "
            "If the doctor asks about information the patient would not know, say you do not know or have not had that test explained to you. "
            "Keep the answer concise and conversational."
        )
        user_prompt = (
            f"Patient-visible case information:\n{patient_context}\n\n"
            f"Patient bias instruction: {_AGENTCLINIC_PATIENT_BIAS_TEXT.get(pb, 'Unsupported patient bias; ignore it.')}\n"
            # f"Doctor bias context: {_AGENTCLINIC_DOCTOR_BIAS_TEXT.get(db, 'Unsupported doctor bias; ignore it.')}\n\n"
            f"Doctor question: {doctor_query}\n\n"
            "Return only the patient's next reply as plain text."
        )
        return system_prompt, user_prompt, bias_notes

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        case_data = _load_case_from_payload(payload, self._runtime_context)
        normalised = AgentClinicCaseAdapter.normalise(case_data)
        doctor_query = str(payload.get("doctor_query", payload.get("question", ""))).strip()
        if not doctor_query:
            raise ValueError("'doctor_query' is required.")

        patient_bias = str(
            payload.get(
                "patient_bias",
                _get_context_value(self._runtime_context, "agentclinic_patient_bias", "patient_bias", default="none"),
            )
        )
        doctor_bias = str(
            payload.get(
                "doctor_bias",
                _get_context_value(self._runtime_context, "agentclinic_doctor_bias", "doctor_bias", default="none"),
            )
        )

        system_prompt, user_prompt, bias_notes = self._build_prompts(
            normalised_case=normalised,
            doctor_query=doctor_query,
            patient_bias=patient_bias,
            doctor_bias=doctor_bias,
        )

        if not self.api_key:
            raise RuntimeError(
                "AgentClinicPatientReplyTool requires an OpenAI-compatible API key. "
                "Set it in configs/global_config.json or OPENAI_API_KEY."
            )

        reply = call_tool_llm(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_seconds,
        )
        reply = _limit_sentences(reply, self.max_sentences)

        return ToolResult(
            name=self.name,
            payload={
                "doctor_query": doctor_query,
                "patient_reply": reply,
                "dataset_variant": normalised["dataset_variant"],
                "simulation_mode": "llm",
                "bias_notes": bias_notes,
            },
            render_fields=["patient_reply"],
        )

    def get_params_schema(self) -> List[ToolParam]:
        return [
            ToolParam(
            name="doctor_query",
            type="string",
            required=True,
            description=(
                "Doctor's current question to the patient. "
                "Keep the question concise, clear, and easy for the patient to answer. "
                "Avoid complex, multi-part, or overly technical questions."
            ),
        ),
        ]

    def get_usage_example(self) -> str:
        return '{"doctor_query": "Can you tell me more about what has been bothering you?"}'


# ===========================================================================
# Tool 2: AgentClinic exam / test request  (LLM-driven)
# ===========================================================================


@dataclass
class AgentClinicRequestExamTool(AgentClinicRuntimeContextMixin):
    name: str = "agentclinic_request_exam"
    description: str = (
        "Call this tool when you need objective clinical data. "
        "Use it to request physical exams, laboratory tests, "
        "or other diagnostic studies and obtain their results."
    )
    api_key: str = ""
    api_base: str = DEFAULT_API_BASE
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 512
    timeout_seconds: int = 60
    _runtime_context: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_base:
            self.api_base = os.environ.get("OPENAI_API_BASE", DEFAULT_API_BASE)

    def _build_prompts(
        self,
        *,
        normalised_case: Dict[str, Any],
        request: str,
    ) -> Tuple[str, str]:
        """Build system and user prompts for the exam result LLM call."""
        # Collect all available exam/test data as a structured text block.
        physical_exam_text = normalised_case.get("physical_exam_text", "").strip()
        test_results_text = normalised_case.get("test_results_text", "").strip()

        available_sections: List[str] = []
        if physical_exam_text:
            available_sections.append(f"Physical Examination Findings:\n{physical_exam_text}")
        if test_results_text:
            available_sections.append(f"Test / Laboratory Results:\n{test_results_text}")

        if available_sections:
            exam_context = "\n\n".join(available_sections)
        else:
            exam_context = "(No examination or test data is available for this case.)"

        # print(exam_context)

        system_prompt = (
            "Your role is to report objective examination and test results to the doctor.\n"
            "Rules:\n"
            "1. Report ONLY findings explicitly present in the provided case data. "
            "Do NOT invent or infer values.\n"
            "2. If the requested examination/test is available, report its findings concisely "
            "in third person.\n"
            "3. If the requested examination/test is not available, but other examinations "
            "were performed, briefly report the available examinations and their findings.\n"
            "4. If no examination data are available at all, reply exactly: "
            "'This examination/test was not performed for this patient.'\n"
            "5. Do NOT provide interpretation, diagnosis, or benchmark metadata."
        )

        user_prompt = (
            f"Available case examination data:\n{exam_context}\n\n"
            f"Doctor's request: {request}\n\n"
            "Report the findings for the requested examination or test. "
            "If the requested test was not performed, report any other available examinations "
            "and their findings."
        )

        print(f"#########exam_context: {exam_context}")
        return system_prompt, user_prompt

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        case_data = _load_case_from_payload(payload, self._runtime_context)
        normalised = AgentClinicCaseAdapter.normalise(case_data)
        request = str(payload.get("request", payload.get("exam_name", ""))).strip()
        if not request:
            raise ValueError("'request' is required.")

        if not self.api_key:
            raise RuntimeError(
                "AgentClinicRequestExamTool requires an OpenAI-compatible API key. "
                "Set it in configs/global_config.json or OPENAI_API_KEY."
            )

        system_prompt, user_prompt = self._build_prompts(
            normalised_case=normalised,
            request=request,
        )

        result_text = call_tool_llm(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            timeout=self.timeout_seconds,
        )

        return ToolResult(
            name=self.name,
            payload={
                "request": request,
                "result_text": result_text.strip(),
                "dataset_variant": normalised["dataset_variant"],
                "simulation_mode": "llm",
            },
            render_fields=["result_text"],
        )

    def get_params_schema(self) -> List[ToolParam]:
        return [
            ToolParam(
                name="request",
                type="string",
                required=True,
                description=(
                    "Requested exam or test name. Use concise, general terms "
                    "(e.g., 'physical examination', 'laboratory results', "
                    "'CT scan', 'vital signs') rather than detailed enumerations."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return '{"request": "neurological examination"}'


# ===========================================================================
# Tool 3: AgentClinic image access
# ===========================================================================


@dataclass
class AgentClinicRequestImageTool(AgentClinicRuntimeContextMixin):
    name: str = "agentclinic_request_image"
    description: str = (
    "Call this tool when medical images are required for diagnosis or analysis. "
    "Retrieve the case image (path or URL) for use with medical image analysis tools."
    )
    image_root_dir: str = ""
    _runtime_context: Dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        case_data = _load_case_from_payload(payload, self._runtime_context)
        normalised = AgentClinicCaseAdapter.normalise(case_data)

        image_path = str(normalised.get("image_path", "")).strip()
        image_url = str(normalised.get("image_url", "")).strip()

        # Prepend root dir when path is relative
        if image_path and self.image_root_dir and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root_dir, image_path)

        has_image = bool(image_path or image_url)

        return ToolResult(
            name=self.name,
            payload={
                "has_image": has_image,
                "image_path": image_path,
                "image_url": image_url,
                "dataset_variant": normalised["dataset_variant"],
                "recommended_next_step": (
                    "Pass 'image_path' to an existing image-analysis tool if a local file is available; otherwise use 'image_url'."
                    if has_image
                    else "This case does not include an image."
                ),
            },
            render_fields=["image_path"],
        )

    def get_params_schema(self) -> List[ToolParam]:
        return []

    def get_usage_example(self) -> str:
        return '{}'
