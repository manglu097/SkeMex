"""Dataset loaders for batch evaluation.

Each loader's process_sample() method returns a standardised dict:
    {
        'id':              sample identifier (str),
        'query':           text query for the agent (str),
        'images':          list of absolute image paths (list[str]),
        'runtime_context': dict for tools, or None,
        'ground_truth':    expected answer (str | list),
        'metadata':        additional info (dict),
    }

Query construction is fully delegated to dataset_prompts.py so that all
prompt logic lives in one place.  Image paths are always appended by the
loader *after* the text query is built (they are passed separately via the
`images` key and handled by AgentLoop / LLM layer).
"""
import json
import os
import hashlib
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional

from medmem_agent.dataset_prompts import (
    build_healthbench_query,
    build_mmmu_query,
    build_mmmu_pro_query,
    build_mediq_query,
    build_agentclinic_text_query,
    build_agentclinic_mm_query,
    build_liveclin_text_query,
    build_liveclin_mm_query,
    build_medxpertqa_text_query,
    build_medxpertqa_mm_query,
    build_medjourney_query,
    build_livemedbbench_query,
)


class DatasetLoader:
    """Base class for dataset loaders."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.dataset_dir = self.data_path.parent

    def load_samples(self) -> List[Dict[str, Any]]:
        """Load and process all samples from the dataset file.

        If the raw JSON line already contains a ``task_category`` field (e.g.
        pre-populated by an offline classification script), it is forwarded
        into the processed sample dict so that ``online.py`` can use it
        directly when ``classifier_mode="precomputed"``.
        """
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    raw = json.loads(line)
                    processed = self.process_sample(raw)
                    # Forward pre-computed task_category if present in raw data.
                    if "task_category" in raw and "task_category" not in processed:
                        processed["task_category"] = raw["task_category"]
                    samples.append(processed)
        return samples

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample into standardized format.

        Subclasses may optionally include a ``task_category`` key in the
        returned dict (list of str).  When absent, the base ``load_samples``
        method will forward the value from the raw JSON line if present.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# HealthBench
# ---------------------------------------------------------------------------
class HealthBenchLoader(DatasetLoader):
    """
    user_input  : sample['prompt'][0]['content']  (no extra suffix)
    ground_truth: sample['rubrics']  — list of rubric dicts
                  (each has 'criterion', 'points', 'tags')
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        prompt_content = sample['prompt'][0]['content']
        query = build_healthbench_query(prompt_content)

        return {
            'id': sample.get('prompt_id', 'unknown'),
            'query': query,
            'images': [],
            'runtime_context': None,
            # ground_truth is the rubrics list (used by the evaluator)
            'ground_truth': sample.get('rubrics', []),
            'metadata': {
                'ideal_completion': (sample.get('ideal_completions_data') or {}).get(
                    'ideal_completion', None
                )
            },
        }


# ---------------------------------------------------------------------------
# MMMU
# ---------------------------------------------------------------------------
class MMMULoader(DatasetLoader):
    """
    user_input  : question + options (A/B/C…) + a Chinese "output only the option letter" suffix
    ground_truth: sample['answer']  — option letter, e.g. 'C'
    images      : sample['image_1'] … sample['image_7'] (if present)
    """

    @property
    def _BASE_PATH(self):
        return Path(os.environ.get("SKEMEX_DATA_ROOT", self.dataset_dir.parent))

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        options_str = sample['options']
        options = (
            ast.literal_eval(options_str)
            if isinstance(options_str, str)
            else options_str
        )

        query = build_mmmu_query(sample['question'], options)

        images = []
        for i in range(1, 8):
            img_key = f'image_{i}'
            if sample.get(img_key):
                images.append(str(self._BASE_PATH / sample[img_key]))

        return {
            'id': sample['id'],
            'query': query,
            'images': images,
            'runtime_context': None,
            'ground_truth': sample['answer'],
            'metadata': {
                'options': options,
                'subfield': sample.get('subfield'),
            },
        }


# ---------------------------------------------------------------------------
# MMMU_Pro
# ---------------------------------------------------------------------------
class MMMUProLoader(DatasetLoader):
    """
    user_input  : question + options (A/B/C…) + a Chinese "output only the option letter" suffix
    ground_truth: sample['answer']  — option letter
    images      : sample['image_1'] … sample['image_7'] (if present)
    """

    @property
    def _BASE_PATH(self):
        return Path(os.environ.get("SKEMEX_DATA_ROOT", self.dataset_dir.parent))

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        options_str = sample['options']
        options = (
            ast.literal_eval(options_str)
            if isinstance(options_str, str)
            else options_str
        )

        query = build_mmmu_pro_query(sample['question'], options)

        images = []
        for i in range(1, 8):
            img_key = f'image_{i}'
            if sample.get(img_key):
                images.append(str(self._BASE_PATH / sample[img_key]))

        return {
            'id': sample['id'],
            'query': query,
            'images': images,
            'runtime_context': None,
            'ground_truth': sample['answer'],
            'metadata': {
                'options': options,
                'subject': sample.get('subject'),
            },
        }


# ---------------------------------------------------------------------------
# MediQ
# ---------------------------------------------------------------------------
class MediQLoader(DatasetLoader):
    """
    user_input  : joined context + question + options + a Chinese "output only the option letter" suffix
    ground_truth: sample['answer_idx']  — option letter, e.g. 'B'
    runtime_context: None — MediQ uses general search/knowledge tools only
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_mediq_query(
            sample['context'], sample['question'], sample['options']
        )

        return {
            'id': sample['id'],
            'query': query,
            'images': [],
            'runtime_context': None,  # MediQ uses general tools only; no specialist tool needed
            'ground_truth': sample['answer_idx'],
            'metadata': {
                'answer_text': sample['answer'],
                'atomic_facts': sample.get('atomic_facts', []),
            },
        }


# ---------------------------------------------------------------------------
# AgentClinic_Text
# ---------------------------------------------------------------------------
class AgentClinicTextLoader(DatasetLoader):
    """
    user_input  : clinical-interview preamble + Objective_for_Doctor
                  + a Chinese "give the most likely diagnosis without explanation" suffix
    ground_truth: sample['OSCE_Examination']['Correct_Diagnosis']
    runtime_context: full OSCE case for tool use
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        osce = sample['OSCE_Examination']
        query = build_agentclinic_text_query(osce['Objective_for_Doctor'])

        return {
            'id': f"agentclinic_text_{hash(str(sample))}",
            'query': query,
            'images': [],
            'runtime_context': {'agentclinic_case': {'OSCE_Examination': osce}},
            'ground_truth': osce['Correct_Diagnosis'],
            'metadata': {
                'patient_demographics': osce['Patient_Actor'].get('Demographics')
            },
        }


# ---------------------------------------------------------------------------
# AgentClinic_MM
# ---------------------------------------------------------------------------
class AgentClinicMMLoader(DatasetLoader):
    """
    user_input  : clinical-interview preamble + question
                  + options (A/B/C… built from sample['answers'])
                  + a Chinese "give the most likely diagnosis; output only the option letter" suffix
    ground_truth: the text of the answer whose 'correct' == True
    images      : sample['image_path']
    runtime_context: patient_info / physical_exams for tool use
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image_path = self.dataset_dir / "images" / sample['image_path']

        runtime_context = {
            'agentclinic_case': {
                'patient_info': sample['patient_info'],
                'physical_exams': sample['physical_exams'],
                'image_path': sample.get('image_path', ''),
            }
        }

        answers = sample['answers']
        # Build query with options from answers list
        query = build_agentclinic_mm_query(sample['question'], answers)

        # Ground truth: letter of the correct option
        correct_letter = None
        for i, ans in enumerate(answers):
            if ans.get('correct'):
                correct_letter = chr(65 + i)
                break

        return {
            'id': sample.get('image_path', 'unknown'),
            'query': query,
            'images': [str(image_path)],
            'runtime_context': runtime_context,
            'ground_truth': correct_letter,
            'metadata': {
                'all_answers': answers,
                'correct_text': next(
                    (a['text'] for a in answers if a.get('correct')), None
                ),
                'type': sample.get('type'),
            },
        }


# ---------------------------------------------------------------------------
# LiveClin_Text
# ---------------------------------------------------------------------------
class LiveClinTextLoader(DatasetLoader):
    """
    user_input  : scenario + question + options + a Chinese "output only the option letter" suffix
    ground_truth: sample['correct_answer']  — option letter
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_liveclin_text_query(
            sample['scenario'], sample['question'], sample['options']
        )

        return {
            'id': sample.get('pmc', 'unknown'),
            'query': query,
            'images': [],
            'runtime_context': None,
            'ground_truth': sample['correct_answer'],
            'metadata': {
                'stage': sample.get('stage'),
                'ICD-10': sample.get('ICD-10'),
            },
        }


# ---------------------------------------------------------------------------
# LiveClin_MM
# ---------------------------------------------------------------------------
class LiveClinMMLoader(DatasetLoader):
    """
    user_input  : scenario + table details + question + options
                  + a Chinese "output only the option letter" suffix
    ground_truth: sample['correct_answer']  — option letter
    images      : scenario_image_details + image_details paths
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_liveclin_mm_query(
            sample['scenario'],
            sample.get('scenario_table_details', []),
            sample['question'],
            sample['options'],
        )

        images = []
        for img in sample.get('scenario_image_details', []):
            if img.get('image_path'):
                images.append(
                    str(self.dataset_dir / "images" / img['image_path'])
                )
        for img in sample.get('image_details', []):
            if img.get('image_path'):
                images.append(
                    str(self.dataset_dir / "images" / img['image_path'])
                )

        return {
            'id': sample.get('pmc', 'unknown'),
            'query': query,
            'images': images,
            'runtime_context': None,
            'ground_truth': sample['correct_answer'],
            'metadata': {
                'stage': sample.get('stage'),
                'ICD-10': sample.get('ICD-10'),
            },
        }


# ---------------------------------------------------------------------------
# MedXpertQA_Text
# ---------------------------------------------------------------------------
class MedXpertQATextLoader(DatasetLoader):
    """
    user_input  : question + options + a Chinese "output only the option letter" suffix
    ground_truth: sample['label']  — option letter
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_medxpertqa_text_query(
            sample['question'], sample['options']
        )

        return {
            'id': sample['id'],
            'query': query,
            'images': [],
            'runtime_context': None,
            'ground_truth': sample['label'],
            'metadata': {
                'medical_task': sample.get('medical_task'),
                'body_system': sample.get('body_system'),
            },
        }


# ---------------------------------------------------------------------------
# MedXpertQA_MM
# ---------------------------------------------------------------------------
class MedXpertQAMMLoader(DatasetLoader):
    """
    user_input  : question + options + a Chinese "output only the option letter" suffix
    ground_truth: sample['label']  — option letter
    images      : sample['image_path'] list
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_medxpertqa_mm_query(
            sample['question'], sample['options']
        )

        images = [
            str(self.dataset_dir / "images" / img_name)
            for img_name in sample.get('image_path', [])
        ]

        return {
            'id': sample['id'],
            'query': query,
            'images': images,
            'runtime_context': None,
            'ground_truth': sample['label'],
            'metadata': {
                'medical_task': sample.get('medical_task'),
                'body_system': sample.get('body_system'),
            },
        }


# ---------------------------------------------------------------------------
# MedJourney
# ---------------------------------------------------------------------------
class MedJourneyLoader(DatasetLoader):
    """
    user_input  : sample['prompt']  (no modification)
    ground_truth: sample['target']
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_medjourney_query(sample['prompt'])

        return {
            'id': f"medjourney_{hash(sample['prompt'][:50])}",
            'query': query,
            'images': [],
            'runtime_context': None,
            'ground_truth': sample['target'],
            'metadata': {'keyentity': sample.get('keyentity')},
        }


# ---------------------------------------------------------------------------
# LiveMedBench
# ---------------------------------------------------------------------------
class LiveMedBenchLoader(DatasetLoader):
    """
    user_input  : narrative + core_request  (no extra suffix)
    ground_truth: sample['rubric_items']  — list of rubric dicts
                  (each has 'criterion' and 'points')
    """

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        query = build_livemedbbench_query(
            sample['narrative'], sample['core_request']
        )

        return {
            'id': sample['case_id'],
            'query': query,
            'images': [],
            'runtime_context': None,
            # ground_truth is the rubric_items list (used by the evaluator)
            'ground_truth': sample.get('rubric_items', []),
            'metadata': {
                'doctor_advice': sample.get('doctor_advice', ''),
            },
        }


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
DATASET_LOADERS = {
    'healthbench': HealthBenchLoader,
    'MMMU': MMMULoader,
    'MMMU_Pro': MMMUProLoader,
    'MediQ': MediQLoader,
    'AgentClinic_Text': AgentClinicTextLoader,
    'AgentClinic_MM': AgentClinicMMLoader,
    'LiveClin_Text': LiveClinTextLoader,
    'LiveClin_MM': LiveClinMMLoader,
    'MedXpertQA_Text': MedXpertQATextLoader,
    'MedXpertQA_MM': MedXpertQAMMLoader,
    'MedJourney': MedJourneyLoader,
    'LiveMedBench': LiveMedBenchLoader,
}
