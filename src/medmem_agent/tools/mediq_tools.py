import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .base import DEFAULT_API_BASE, ToolParam, ToolResult, call_tool_llm
from .agentclinic_tools import _limit_sentences

logger = logging.getLogger(__name__)

def _mediq_context(runtime_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(runtime_context, dict):
        return {}
    nested = runtime_context.get("mediq")
    if isinstance(nested, dict):
        return nested
    return runtime_context

def _get_context_value(runtime_context: Optional[Dict[str, Any]], *keys: str, default: Any = None) -> Any:
    context = _mediq_context(runtime_context)
    for key in keys:
        if key in context and context[key] not in (None, ""):
            return context[key]
    return default

def _load_case_from_payload(payload: Dict[str, Any], runtime_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    case_data = payload.get("case_data")
    if case_data is None:
        case_data = _get_context_value(
            runtime_context,
            "mediq_case",
            "case_data",
            "current_case",
        )

    if not isinstance(case_data, dict):
        raise ValueError(
            "No MediQ case is bound to this tool call. "
            "Bind the current case through ToolRegistry.set_runtime_context({'mediq_case': case_dict}) "
            "or AgentLoop.run(..., tool_runtime_context={'mediq_case': case_dict})."
        )
    return case_data

class MediQRuntimeContextMixin:
    def set_runtime_context(self, context: Optional[Dict[str, Any]]) -> None:
        self._runtime_context = dict(context or {})

@dataclass
class MediQPatientReplyTool(MediQRuntimeContextMixin):
    name: str = "mediq_patient_reply"
    description: str = (
        "Call this tool when additional patient information is needed. "
        "Ask the patient questions to gather symptoms, medical history, "
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
        case_data: Dict[str, Any],
        doctor_query: str,
    ) -> Tuple[str, str]:
        patient_info = case_data.get("patient", {})
        facts_old = case_data.get("facts_old", [])
        atomic_facts = case_data.get("atomic_facts", [])

        patient_desc = f"Gender: {patient_info.get('gender', 'Unknown')}, Age: {patient_info.get('age', 'Unknown')}"
        
        facts_text = "\n".join(facts_old) if isinstance(facts_old, list) else str(facts_old)
        atomic_text = "\n".join(atomic_facts) if isinstance(atomic_facts, list) else str(atomic_facts)

        patient_context = (
            f"Patient Demographics:\n{patient_desc}\n\n"
            f"Case Facts:\n{facts_text}\n\n"
            f"Detailed Atomic Facts:\n{atomic_text}"
        )

        system_prompt = (
            "You are role-playing the patient in a MediQ clinical benchmark. "
            "Answer the doctor's question naturally in first person, as the patient only. "
            "Use ONLY the patient-visible information provided in the facts. "
            "Do NOT invent new symptoms, timing, exposures, medications, diagnoses, exam findings, lab values, imaging results, or treatments. "
            "Do NOT reveal hidden benchmark labels or the correct diagnosis. "
            "If the doctor asks about information the patient would not know, say you do not know or have not had that test explained to you. "
            "Keep the answer concise and conversational."
        )
        user_prompt = (
            f"Patient-visible case information:\n{patient_context}\n\n"
            f"Doctor question: {doctor_query}\n\n"
            "Return only the patient's next reply as plain text."
        )
        return system_prompt, user_prompt

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        case_data = _load_case_from_payload(payload, self._runtime_context)
        doctor_query = str(payload.get("doctor_query", payload.get("question", ""))).strip()
        if not doctor_query:
            raise ValueError("'doctor_query' is required.")

        system_prompt, user_prompt = self._build_prompts(
            case_data=case_data,
            doctor_query=doctor_query,
        )

        if not self.api_key:
            raise RuntimeError(
                "MediQPatientReplyTool requires an OpenAI-compatible API key. "
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
                "simulation_mode": "llm",
            },
            render_fields=["patient_reply"],
        )

    def get_params_schema(self) -> List[ToolParam]:
        return [
            ToolParam(
                name="doctor_query",
                type="string",
                required=True,
                description="The doctor's current question or utterance to the patient. The current MediQ case is supplied by the benchmark runtime context, not by the model.",
            ),
        ]

    def get_usage_example(self) -> str:
        return '{"doctor_query": "Can you tell me more about what has been bothering you?"}'
