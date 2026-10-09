"""Medical calculation tools.

This module provides three tools for quantitative reasoning in clinical contexts:

  1. PythonCalculatorTool       — Safe Python expression evaluator for
                                  arbitrary mathematical calculations.
  2. MedicalUnitConverterTool   — Convert between common medical units
                                  (lab values, weight, temperature, etc.).
  3. ClinicalScoreCalculatorTool — Compute widely-used clinical risk scores
                                   (CHA₂DS₂-VASc, APACHE II, Child-Pugh,
                                    Wells DVT/PE, CURB-65, GCS, BMI, etc.)

Design principle
-----------------
All tools are **pure Python with zero external dependencies** so they work
in any environment without network access.
"""

from __future__ import annotations

import ast
import logging
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .base import ToolParam, ToolResult

logger = logging.getLogger(__name__)


# ===========================================================================
# Safe expression evaluator (whitelist-based AST eval)
# ===========================================================================
_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "pi": math.pi,
    "e": math.e,
    "pow": math.pow,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate a whitelisted AST node."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCTIONS:
            return _SAFE_FUNCTIONS[node.id]
        raise ValueError(f"Undefined name: '{node.id}'")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return _SAFE_OPERATORS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return _SAFE_OPERATORS[op_type](operand)
    if isinstance(node, ast.Call):
        func = _safe_eval(node.func)
        if not callable(func):
            raise ValueError(f"Not callable: {func}")
        args = [_safe_eval(a) for a in node.args]
        return func(*args)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


# ===========================================================================
# PythonCalculatorTool
# ===========================================================================
@dataclass
class PythonCalculatorTool:
    """Safe Python expression evaluator for medical calculations.

    Evaluates mathematical expressions using a whitelist-based AST evaluator.
    Supports arithmetic operators (+, -, *, /, //, %, **), common math
    functions (sqrt, log, exp, ceil, floor, abs, round), and constants
    (pi, e). No arbitrary code execution is possible.

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
    """

    name: str = "python_calculator"
    description: str = (
        "Evaluate a mathematical expression precisely. Supports arithmetic, common math functions "
        "(sqrt, log, exp, etc.), and constants (pi, e). Use for dosage calculations or any "
        "quantitative reasoning."
    )

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Evaluate a mathematical expression.

        Args:
            payload: Must contain ``"expression"`` (str): the mathematical
                expression to evaluate. Optionally ``"context"`` (str): a
                human-readable description of what is being calculated.

        Returns:
            ToolResult with keys:
                - ``expression``: the input expression.
                - ``result``: the numerical result.
                - ``result_str``: formatted result string.
                - ``context``: the provided context (if any).

        Raises:
            ValueError: If ``expression`` is missing, empty, or contains
                unsafe constructs.
        """
        expression = str(payload.get("expression", "")).strip()
        if not expression:
            raise ValueError("'expression' field is required and must not be empty.")

        context = str(payload.get("context", "")).strip()

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
        except ZeroDivisionError:
            raise ValueError("Division by zero in expression.")
        except (ValueError, TypeError) as e:
            raise ValueError(f"Expression evaluation error: {e}")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {e}")

        # Format result
        if isinstance(result, float):
            if result == int(result) and abs(result) < 1e15:
                result_str = str(int(result))
            else:
                result_str = f"{result:.6g}"
        else:
            result_str = str(result)

        out: Dict[str, Any] = {
            "expression": expression,
            "result": result,
            "result_str": result_str,
        }
        if context:
            out["context"] = context

        return ToolResult(name=self.name, payload=out, render_fields=["result"])

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="expression",
                type="string",
                required=True,
                description=(
                    "A mathematical expression to evaluate. Supports: arithmetic (+, -, *, /, "
                    "**, //), functions (sqrt, log, log10, exp, ceil, floor, abs, round, min, "
                    "max), and constants (pi, e). Example: \"70 * 0.5\" or \"sqrt(144)\". "
                    "Do NOT include variable names or assignment statements."
                ),
            ),
            ToolParam(
                name="context",
                type="string",
                required=False,
                description=(
                    "Human-readable description of what is being calculated. "
                    "Example: \"Dose calculation: 0.5 mg/kg for 70 kg patient\"."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"expression": "70 * 0.5", "context": "Dose calculation: 0.5 mg/kg for 70 kg patient"}'
        )


# ===========================================================================
# MedicalUnitConverterTool
# ===========================================================================

# Conversion table: (from_unit, to_unit) → factor or callable
# All conversions are to SI base unit first, then to target.
# Format: canonical_name → (SI_factor, SI_unit)
_UNIT_REGISTRY: Dict[str, Tuple[float, str]] = {
    # Mass
    "kg": (1.0, "kg"),
    "g": (0.001, "kg"),
    "mg": (1e-6, "kg"),
    "mcg": (1e-9, "kg"),
    "ug": (1e-9, "kg"),
    "lb": (0.453592, "kg"),
    "lbs": (0.453592, "kg"),
    "oz": (0.0283495, "kg"),
    # Length
    "m": (1.0, "m"),
    "cm": (0.01, "m"),
    "mm": (0.001, "m"),
    "in": (0.0254, "m"),
    "ft": (0.3048, "m"),
    # Volume
    "l": (1.0, "L"),
    "dl": (0.1, "L"),
    "ml": (0.001, "L"),
    "ul": (1e-6, "L"),
    "fl oz": (0.0295735, "L"),
    # Concentration (mass/volume)
    "mg/dl": (0.01, "g/L"),
    "mg/l": (0.001, "g/L"),
    "g/dl": (10.0, "g/L"),
    "g/l": (1.0, "g/L"),
    "ug/dl": (0.00001, "g/L"),
    "ug/ml": (0.001, "g/L"),
    # Pressure
    "mmhg": (1.0, "mmHg"),
    "kpa": (7.50062, "mmHg"),
    "cmh2o": (0.735559, "mmHg"),
    "atm": (760.0, "mmHg"),
    # Temperature (handled separately)
    "c": ("temp", "°C"),
    "°c": ("temp", "°C"),
    "celsius": ("temp", "°C"),
    "f": ("temp", "°F"),
    "°f": ("temp", "°F"),
    "fahrenheit": ("temp", "°F"),
    "k": ("temp", "K"),
    "kelvin": ("temp", "K"),
    # Energy (nutrition support)
    "kcal": (1.0, "kcal"),
    "cal": (0.001, "kcal"),
    "kj": (0.239006, "kcal"),
    "j": (0.000239006, "kcal"),
    # Time
    "hr": (1.0, "hr"),
    "hour": (1.0, "hr"),
    "hours": (1.0, "hr"),
    "min": (1 / 60, "hr"),
    "minute": (1 / 60, "hr"),
    "minutes": (1 / 60, "hr"),
    "sec": (1 / 3600, "hr"),
    "second": (1 / 3600, "hr"),
    "seconds": (1 / 3600, "hr"),
    # Flow rate (IV infusion)
    "ml/hr": (1.0, "mL/hr"),
    "ml/h": (1.0, "mL/hr"),
    "ml/min": (60.0, "mL/hr"),
    "l/hr": (1000.0, "mL/hr"),
    "l/h": (1000.0, "mL/hr"),
    "l/min": (60000.0, "mL/hr"),
    # HbA1c (handled separately via "hba1c" special path)
    "hba1c_%": ("hba1c", "%"),
    "hba1c_mmol/mol": ("hba1c", "mmol/mol"),
    "ngsp": ("hba1c", "%"),
    "ifcc": ("hba1c", "mmol/mol"),
}

# Lab-specific molar conversions: substance → (MW in g/mol)
_MOLAR_WEIGHTS: Dict[str, float] = {
    "glucose": 180.16,
    "cholesterol": 386.65,
    "triglycerides": 885.43,
    "creatinine": 113.12,
    "urea": 60.06,
    "bun": 28.014,  # blood urea nitrogen (N component of urea)
    "sodium": 22.99,
    "potassium": 39.10,
    "calcium": 40.08,
    "magnesium": 24.31,
    "phosphorus": 30.97,
    "phosphate": 30.97,
    "hemoglobin": 64500.0,
    "albumin": 66500.0,
    "bilirubin": 584.66,
    "uric acid": 168.11,
    "uric_acid": 168.11,
    "iron": 55.845,
    "lithium": 6.941,
    "chloride": 35.45,
    "bicarbonate": 61.02,
    "hco3": 61.02,
    "ldl": 386.65,  # same MW as cholesterol
    "hdl": 386.65,
    "hba1c": None,  # percentage, not molar
}

# Electrolytes with valence for mEq/L <-> mmol/L conversion
# mEq/L = mmol/L * valence
_ELECTROLYTE_VALENCE: Dict[str, int] = {
    "sodium": 1,
    "na": 1,
    "potassium": 1,
    "k": 1,
    "chloride": 1,
    "cl": 1,
    "bicarbonate": 1,
    "hco3": 1,
    "calcium": 2,
    "ca": 2,
    "magnesium": 2,
    "mg": 2,
    "phosphate": 3,
    "phosphorus": 3,
    "p": 3,
}


def _convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between temperature units."""
    from_unit = from_unit.lower().replace("°", "").replace("celsius", "c").replace("fahrenheit", "f").replace("kelvin", "k")
    to_unit = to_unit.lower().replace("°", "").replace("celsius", "c").replace("fahrenheit", "f").replace("kelvin", "k")

    # Convert to Celsius first
    if from_unit in ("f",):
        celsius = (value - 32) * 5 / 9
    elif from_unit in ("k",):
        celsius = value - 273.15
    else:
        celsius = value

    # Convert from Celsius to target
    if to_unit in ("f",):
        return celsius * 9 / 5 + 32
    elif to_unit in ("k",):
        return celsius + 273.15
    else:
        return celsius


@dataclass
class MedicalUnitConverterTool:
    """Convert between common medical and laboratory units.

    Handles mass, length, volume, concentration, pressure, and temperature.
    Also supports molar ↔ mass conversions for common lab analytes
    (e.g., mmol/L ↔ mg/dL for glucose, creatinine, cholesterol, etc.).

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
    """

    name: str = "unit_converter"
    description: str = (
        "Convert between medical/lab units: mass, length, volume, concentration, pressure, "
        "temperature, molar (mmol/L ↔ mg/dL), electrolytes (mEq/L ↔ mmol/L), and HbA1c."
    )

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Convert a value between units.

        Args:
            payload: Must contain:
                - ``"value"`` (float): the numerical value to convert.
                - ``"from_unit"`` (str): the source unit.
                - ``"to_unit"`` (str): the target unit.
                Optionally ``"substance"`` (str): the analyte name for molar
                conversions (e.g., ``"glucose"``).

        Returns:
            ToolResult with keys:
                - ``input``: original value and unit.
                - ``output``: converted value and unit.
                - ``conversion_factor``: the factor applied.

        Raises:
            ValueError: If required fields are missing or units are unsupported.
        """
        try:
            value = float(payload.get("value", ""))
        except (TypeError, ValueError):
            raise ValueError("'value' must be a number.")

        from_unit = str(payload.get("from_unit", "")).strip().lower()
        to_unit = str(payload.get("to_unit", "")).strip().lower()
        substance = str(payload.get("substance", "")).strip().lower()

        if not from_unit or not to_unit:
            raise ValueError("'from_unit' and 'to_unit' are required.")

        # ---- Temperature (special case) ----
        from_entry = _UNIT_REGISTRY.get(from_unit)
        to_entry = _UNIT_REGISTRY.get(to_unit)

        if from_entry and from_entry[0] == "temp":
            converted = _convert_temperature(value, from_unit, to_unit)
            return ToolResult(
                name=self.name,
                payload={
                    "input": f"{value} {from_unit}",
                    "output": f"{converted:.4g} {to_unit}",
                    "conversion_factor": "temperature formula",
                },
                render_fields=["output"],
            )

        # ---- HbA1c: NGSP (%) ↔ IFCC (mmol/mol) ----
        if from_entry and from_entry[0] == "hba1c":
            if to_entry is None or to_entry[0] != "hba1c":
                raise ValueError(
                    "HbA1c can only be converted between '%' (NGSP) and 'mmol/mol' (IFCC). "
                    f"Use 'hba1c_%' or 'ngsp' for NGSP, 'hba1c_mmol/mol' or 'ifcc' for IFCC."
                )
            from_display = from_entry[1]
            to_display = to_entry[1]
            if from_display == "%" and to_display == "mmol/mol":
                # NGSP → IFCC: (% − 2.15) × 10.929  (IFCC formula)
                converted = (value - 2.15) * 10.929
                formula = "(NGSP% − 2.15) × 10.929"
            elif from_display == "mmol/mol" and to_display == "%":
                # IFCC → NGSP: (mmol/mol / 10.929) + 2.15
                converted = value / 10.929 + 2.15
                formula = "(IFCC mmol/mol ÷ 10.929) + 2.15"
            else:
                converted = value
                formula = "identity"
            return ToolResult(
                name=self.name,
                payload={
                    "input": f"{value} {from_display} (HbA1c)",
                    "output": f"{converted:.2f} {to_display}",
                    "conversion_factor": formula,
                    "note": "IFCC–NGSP master equation (IFCC 2002)",
                },
                render_fields=["output"],
            )

        # ---- mEq/L ↔ mmol/L (electrolytes) ----
        meql_pattern = re.compile(r"meq/l", re.IGNORECASE)
        mmol_l_pattern = re.compile(r"mmol/l", re.IGNORECASE)
        if substance and substance in _ELECTROLYTE_VALENCE:
            valence = _ELECTROLYTE_VALENCE[substance]
            if meql_pattern.match(from_unit) and mmol_l_pattern.match(to_unit):
                converted = value / valence
                factor = 1.0 / valence
                return ToolResult(
                    name=self.name,
                    payload={
                        "input": f"{value} {from_unit} ({substance})",
                        "output": f"{converted:.4g} {to_unit}",
                        "conversion_factor": f"{factor:.4g}",
                        "note": f"Valence of {substance} = {valence}; mEq/L ÷ valence = mmol/L",
                    },
                    render_fields=["output"],
                )
            elif mmol_l_pattern.match(from_unit) and meql_pattern.match(to_unit):
                converted = value * valence
                factor = float(valence)
                return ToolResult(
                    name=self.name,
                    payload={
                        "input": f"{value} {from_unit} ({substance})",
                        "output": f"{converted:.4g} {to_unit}",
                        "conversion_factor": f"{factor:.4g}",
                        "note": f"Valence of {substance} = {valence}; mmol/L × valence = mEq/L",
                    },
                    render_fields=["output"],
                )
            # else: fall through to molar MW path or standard path

        # ---- Molar ↔ mass conversion (e.g., mmol/L ↔ mg/dL) ----
        mmol_pattern = re.compile(r"mmol/l", re.IGNORECASE)
        mgdl_pattern = re.compile(r"mg/dl", re.IGNORECASE)

        if substance and substance in _MOLAR_WEIGHTS:
            mw = _MOLAR_WEIGHTS[substance]
            if mw is None:
                raise ValueError(f"Molar conversion not applicable for '{substance}'.")

            if mmol_pattern.match(from_unit) and mgdl_pattern.match(to_unit):
                # mmol/L → mg/dL: multiply by MW / 10
                converted = value * mw / 10
                factor = mw / 10
            elif mgdl_pattern.match(from_unit) and mmol_pattern.match(to_unit):
                # mg/dL → mmol/L: multiply by 10 / MW
                converted = value * 10 / mw
                factor = 10 / mw
            else:
                raise ValueError(
                    f"Molar conversion only supported between mmol/L and mg/dL. "
                    f"Got: {from_unit} → {to_unit}."
                )
            return ToolResult(
                name=self.name,
                payload={
                    "input": f"{value} {from_unit} ({substance})",
                    "output": f"{converted:.4g} {to_unit}",
                    "conversion_factor": f"{factor:.4g}",
                    "molecular_weight": f"{mw} g/mol",
                },
                render_fields=["output"],
            )

        # ---- Standard unit conversion via SI base ----
        if from_entry is None:
            raise ValueError(
                f"Unsupported source unit: '{from_unit}'. "
                f"Supported units: {sorted(_UNIT_REGISTRY.keys())}"
            )
        if to_entry is None:
            raise ValueError(
                f"Unsupported target unit: '{to_unit}'. "
                f"Supported units: {sorted(_UNIT_REGISTRY.keys())}"
            )

        from_factor, from_si = from_entry
        to_factor, to_si = to_entry

        if from_si != to_si:
            raise ValueError(
                f"Cannot convert between incompatible unit types: "
                f"'{from_unit}' ({from_si}) and '{to_unit}' ({to_si}). "
                "For molar conversions, provide 'substance' (e.g., 'glucose')."
            )

        converted = value * from_factor / to_factor
        factor = from_factor / to_factor

        return ToolResult(
            name=self.name,
            payload={
                "input": f"{value} {from_unit}",
                "output": f"{converted:.6g} {to_unit}",
                "conversion_factor": f"{factor:.6g}",
            },
            render_fields=["output"],
        )

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            ToolParam(
                name="value",
                type="number",
                required=True,
                description="The numerical value to convert.",
            ),
            ToolParam(
                name="from_unit",
                type="string",
                required=True,
                description=(
                    "Source unit. Supported: mass (kg, g, mg, mcg, lb, oz), "
                    "length (m, cm, mm, in, ft), volume (L, dL, mL, uL), "
                    "pressure (mmHg, kPa, cmH2O), temperature (C, F, K), "
                    "concentration (mg/dL, mmol/L, g/L, g/dL), "
                    "energy (kcal, kJ), flow rate (mL/hr, mL/min, L/hr), "
                    "time (hr, min, sec), "
                    "HbA1c (hba1c_%, ngsp, hba1c_mmol/mol, ifcc), "
                    "electrolytes mEq/L (meq/L with substance=sodium/potassium/calcium/etc). "
                    "Case-insensitive."
                ),
            ),
            ToolParam(
                name="to_unit",
                type="string",
                required=True,
                description="Target unit. Must be in the same category as 'from_unit'.",
            ),
            ToolParam(
                name="substance",
                type="string",
                required=False,
                description=(
                    "Analyte or electrolyte name for molar/valence conversions. "
                    "For mmol/L ↔ mg/dL: glucose, creatinine, cholesterol, urea, bun, "
                    "sodium, potassium, calcium, magnesium, phosphate, bilirubin, "
                    "uric_acid, triglycerides, iron, lithium, chloride, bicarbonate, ldl, hdl. "
                    "For mEq/L ↔ mmol/L: sodium, potassium, calcium, magnesium, "
                    "chloride, bicarbonate (hco3), phosphate."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"value": 5.5, "from_unit": "mmol/L", "to_unit": "mg/dL", "substance": "glucose"}'
        )


# ===========================================================================
# ClinicalScoreCalculatorTool
# ===========================================================================

def _score_cha2ds2_vasc(params: Dict[str, Any]) -> Dict[str, Any]:
    """CHA₂DS₂-VASc score for AF stroke risk."""
    score = 0
    breakdown = []

    if params.get("chf"):
        score += 1; breakdown.append("Congestive heart failure: +1")
    if params.get("hypertension"):
        score += 1; breakdown.append("Hypertension: +1")
    age = int(params.get("age", 0))
    if age >= 75:
        score += 2; breakdown.append("Age ≥75: +2")
    elif age >= 65:
        score += 1; breakdown.append("Age 65–74: +1")
    if params.get("diabetes"):
        score += 1; breakdown.append("Diabetes mellitus: +1")
    if params.get("stroke") or params.get("tia"):
        score += 2; breakdown.append("Prior stroke/TIA: +2")
    if params.get("vascular_disease"):
        score += 1; breakdown.append("Vascular disease: +1")
    if str(params.get("sex", "")).lower() in ("f", "female"):
        score += 1; breakdown.append("Female sex: +1")

    risk_map = {0: "Low (<1%/yr)", 1: "Low-moderate (~1.3%/yr)",
                2: "Moderate (~2.2%/yr)", 3: "High (~3.2%/yr)",
                4: "High (~4.0%/yr)", 5: "High (~6.7%/yr)",
                6: "High (~9.8%/yr)", 7: "High (~9.6%/yr)",
                8: "High (~12.5%/yr)", 9: "High (~15.2%/yr)"}
    risk = risk_map.get(score, "High")
    anticoag = "Anticoagulation recommended" if score >= 2 else (
        "Consider anticoagulation" if score == 1 else "No anticoagulation needed"
    )
    return {
        "score": score,
        "stroke_risk": risk,
        "recommendation": anticoag,
        "breakdown": breakdown,
    }


def _score_wells_dvt(params: Dict[str, Any]) -> Dict[str, Any]:
    """Wells DVT score."""
    score = 0
    breakdown = []

    criteria = [
        ("active_cancer", 1, "Active cancer: +1"),
        ("paralysis_paresis", 1, "Paralysis/paresis/immobilisation: +1"),
        ("bedridden_3d", 1, "Bedridden >3 days or surgery within 4 weeks: +1"),
        ("tenderness_deep_vein", 1, "Localised tenderness along deep vein: +1"),
        ("entire_leg_swollen", 1, "Entire leg swollen: +1"),
        ("calf_swelling_3cm", 1, "Calf swelling >3 cm vs asymptomatic side: +1"),
        ("pitting_oedema", 1, "Pitting oedema (symptomatic leg only): +1"),
        ("collateral_veins", 1, "Collateral superficial veins: +1"),
        ("previous_dvt", 1, "Previously documented DVT: +1"),
    ]
    for key, pts, label in criteria:
        if params.get(key):
            score += pts; breakdown.append(label)

    if params.get("alternative_diagnosis"):
        score -= 2; breakdown.append("Alternative diagnosis as likely: -2")

    if score >= 2:
        risk = "High (DVT likely)"
    elif score == 1:
        risk = "Moderate"
    else:
        risk = "Low (DVT unlikely)"

    return {"score": score, "probability": risk, "breakdown": breakdown}


def _score_wells_pe(params: Dict[str, Any]) -> Dict[str, Any]:
    """Wells PE score."""
    score = 0.0
    breakdown = []

    if params.get("clinical_signs_dvt"):
        score += 3; breakdown.append("Clinical signs of DVT: +3")
    if params.get("pe_most_likely"):
        score += 3; breakdown.append("PE is most likely diagnosis: +3")
    if params.get("heart_rate_over_100"):
        score += 1.5; breakdown.append("Heart rate >100 bpm: +1.5")
    if params.get("immobilisation_surgery"):
        score += 1.5; breakdown.append("Immobilisation/surgery in past 4 weeks: +1.5")
    if params.get("previous_dvt_pe"):
        score += 1.5; breakdown.append("Previous DVT/PE: +1.5")
    if params.get("haemoptysis"):
        score += 1; breakdown.append("Haemoptysis: +1")
    if params.get("malignancy"):
        score += 1; breakdown.append("Malignancy: +1")

    if score > 6:
        risk = "High probability (>60%)"
    elif score > 2:
        risk = "Moderate probability (20–60%)"
    else:
        risk = "Low probability (<10%)"

    return {"score": score, "probability": risk, "breakdown": breakdown}


def _score_curb65(params: Dict[str, Any]) -> Dict[str, Any]:
    """CURB-65 pneumonia severity score."""
    score = 0
    breakdown = []

    if params.get("confusion"):
        score += 1; breakdown.append("Confusion (new): +1")
    bun = float(params.get("bun", 0))
    if bun > 19:
        score += 1; breakdown.append(f"BUN >19 mg/dL ({bun}): +1")
    rr = int(params.get("respiratory_rate", 0))
    if rr >= 30:
        score += 1; breakdown.append(f"Respiratory rate ≥30 ({rr}): +1")
    sbp = int(params.get("sbp", 999))
    dbp = int(params.get("dbp", 999))
    if sbp < 90 or dbp <= 60:
        score += 1; breakdown.append(f"BP <90/60 ({sbp}/{dbp}): +1")
    age = int(params.get("age", 0))
    if age >= 65:
        score += 1; breakdown.append(f"Age ≥65 ({age}): +1")

    severity_map = {
        0: "Low (30-day mortality ~0.6%) — outpatient treatment",
        1: "Low (30-day mortality ~2.7%) — outpatient treatment",
        2: "Moderate (30-day mortality ~6.8%) — consider hospitalisation",
        3: "Severe (30-day mortality ~14%) — hospitalise",
        4: "Severe (30-day mortality ~27.8%) — consider ICU",
        5: "Severe (30-day mortality ~27.8%) — consider ICU",
    }
    severity = severity_map.get(score, "Severe — ICU")
    return {"score": score, "severity": severity, "breakdown": breakdown}


def _score_child_pugh(params: Dict[str, Any]) -> Dict[str, Any]:
    """Child-Pugh score for liver cirrhosis severity."""
    score = 0
    breakdown = []

    # Bilirubin (mg/dL)
    bili = float(params.get("bilirubin", 0))
    if bili < 2:
        score += 1; breakdown.append(f"Bilirubin <2 mg/dL: +1")
    elif bili <= 3:
        score += 2; breakdown.append(f"Bilirubin 2–3 mg/dL: +2")
    else:
        score += 3; breakdown.append(f"Bilirubin >3 mg/dL: +3")

    # Albumin (g/dL)
    alb = float(params.get("albumin", 0))
    if alb > 3.5:
        score += 1; breakdown.append(f"Albumin >3.5 g/dL: +1")
    elif alb >= 2.8:
        score += 2; breakdown.append(f"Albumin 2.8–3.5 g/dL: +2")
    else:
        score += 3; breakdown.append(f"Albumin <2.8 g/dL: +3")

    # INR
    inr = float(params.get("inr", 0))
    if inr < 1.7:
        score += 1; breakdown.append(f"INR <1.7: +1")
    elif inr <= 2.3:
        score += 2; breakdown.append(f"INR 1.7–2.3: +2")
    else:
        score += 3; breakdown.append(f"INR >2.3: +3")

    # Ascites
    ascites = str(params.get("ascites", "none")).lower()
    if ascites == "none":
        score += 1; breakdown.append("No ascites: +1")
    elif ascites == "mild":
        score += 2; breakdown.append("Mild ascites: +2")
    else:
        score += 3; breakdown.append("Moderate/severe ascites: +3")

    # Encephalopathy
    enceph = int(params.get("encephalopathy_grade", 0))
    if enceph == 0:
        score += 1; breakdown.append("No encephalopathy: +1")
    elif enceph in (1, 2):
        score += 2; breakdown.append(f"Grade {enceph} encephalopathy: +2")
    else:
        score += 3; breakdown.append(f"Grade {enceph} encephalopathy: +3")

    if score <= 6:
        cls = "Class A (5–6): Well-compensated, 1-yr survival ~100%"
    elif score <= 9:
        cls = "Class B (7–9): Significant functional compromise, 1-yr survival ~80%"
    else:
        cls = "Class C (10–15): Decompensated, 1-yr survival ~45%"

    return {"score": score, "class": cls, "breakdown": breakdown}


def _score_gcs(params: Dict[str, Any]) -> Dict[str, Any]:
    """Glasgow Coma Scale."""
    eye = int(params.get("eye_opening", 0))
    verbal = int(params.get("verbal_response", 0))
    motor = int(params.get("motor_response", 0))

    eye = max(1, min(eye, 4))
    verbal = max(1, min(verbal, 5))
    motor = max(1, min(motor, 6))

    total = eye + verbal + motor
    if total >= 13:
        severity = "Mild TBI"
    elif total >= 9:
        severity = "Moderate TBI"
    else:
        severity = "Severe TBI"

    return {
        "score": total,
        "eye_opening": eye,
        "verbal_response": verbal,
        "motor_response": motor,
        "severity": severity,
        "breakdown": [
            f"Eye opening: {eye}/4",
            f"Verbal response: {verbal}/5",
            f"Motor response: {motor}/6",
        ],
    }


def _score_bmi(params: Dict[str, Any]) -> Dict[str, Any]:
    """Body Mass Index."""
    weight_kg = float(params.get("weight_kg", 0))
    height_m = float(params.get("height_m", 0))
    if height_m <= 0:
        height_cm = float(params.get("height_cm", 0))
        height_m = height_cm / 100.0
    if weight_kg <= 0 or height_m <= 0:
        raise ValueError("'weight_kg' and 'height_m' (or 'height_cm') are required.")

    bmi = weight_kg / (height_m ** 2)
    if bmi < 18.5:
        category = "Underweight"
    elif bmi < 25:
        category = "Normal weight"
    elif bmi < 30:
        category = "Overweight"
    elif bmi < 35:
        category = "Obese class I"
    elif bmi < 40:
        category = "Obese class II"
    else:
        category = "Obese class III (morbid)"

    return {
        "bmi": round(bmi, 2),
        "category": category,
        "weight_kg": weight_kg,
        "height_m": round(height_m, 3),
    }


def _score_egfr_ckd_epi(params: Dict[str, Any]) -> Dict[str, Any]:
    """eGFR using CKD-EPI 2021 creatinine equation."""
    scr = float(params.get("creatinine_mgdl", 0))
    age = int(params.get("age", 0))
    sex = str(params.get("sex", "male")).lower()

    if scr <= 0 or age <= 0:
        raise ValueError("'creatinine_mgdl' and 'age' are required.")

    # CKD-EPI 2021 (race-free)
    if sex in ("f", "female"):
        kappa, alpha = 0.7, -0.241
        sex_factor = 1.012
    else:
        kappa, alpha = 0.9, -0.302
        sex_factor = 1.0

    ratio = scr / kappa
    if ratio < 1:
        egfr = 142 * (ratio ** alpha) * (0.9938 ** age) * sex_factor
    else:
        egfr = 142 * (ratio ** -1.200) * (0.9938 ** age) * sex_factor

    if egfr >= 90:
        stage = "G1 (Normal or high, ≥90)"
    elif egfr >= 60:
        stage = "G2 (Mildly decreased, 60–89)"
    elif egfr >= 45:
        stage = "G3a (Mildly to moderately decreased, 45–59)"
    elif egfr >= 30:
        stage = "G3b (Moderately to severely decreased, 30–44)"
    elif egfr >= 15:
        stage = "G4 (Severely decreased, 15–29)"
    else:
        stage = "G5 (Kidney failure, <15)"

    return {
        "egfr_ml_min_1.73m2": round(egfr, 1),
        "ckd_stage": stage,
        "creatinine_mgdl": scr,
        "age": age,
        "sex": sex,
    }


def _score_has_bled(params: Dict[str, Any]) -> Dict[str, Any]:
    """HAS-BLED score for major bleeding risk in anticoagulated AF patients."""
    score = 0
    breakdown = []
    if params.get("hypertension"):  # uncontrolled SBP >160 mmHg
        score += 1; breakdown.append("Hypertension (uncontrolled SBP >160): +1")
    if params.get("renal_disease"):  # dialysis, transplant, Cr >2.26 mg/dL
        score += 1; breakdown.append("Renal disease: +1")
    if params.get("liver_disease"):  # cirrhosis or bilirubin >2×ULN + AST/ALT/ALP >3×ULN
        score += 1; breakdown.append("Liver disease: +1")
    if params.get("stroke_history"):
        score += 1; breakdown.append("Stroke history: +1")
    if params.get("bleeding_history") or params.get("bleeding_predisposition"):
        score += 1; breakdown.append("Prior bleeding or predisposition: +1")
    if params.get("labile_inr"):  # TTR <60%
        score += 1; breakdown.append("Labile INR (TTR <60%): +1")
    age = int(params.get("age", 0))
    if age >= 65:
        score += 1; breakdown.append(f"Age ≥65 ({age}): +1")
    if params.get("antiplatelet_nsaid"):  # aspirin, clopidogrel, NSAID
        score += 1; breakdown.append("Antiplatelet/NSAID use: +1")
    if params.get("alcohol"):  # ≥8 drinks/week
        score += 1; breakdown.append("Alcohol use (≥8 drinks/week): +1")
    if score >= 3:
        risk = "High (≥3): increased bleeding risk — caution with anticoagulation"
    elif score == 2:
        risk = "Moderate (2): moderate bleeding risk"
    else:
        risk = "Low (0–1): low bleeding risk"
    return {"score": score, "bleeding_risk": risk, "breakdown": breakdown}


def _score_heart(params: Dict[str, Any]) -> Dict[str, Any]:
    """HEART score for major adverse cardiac events (MACE) in chest pain."""
    score = 0
    breakdown = []
    # History
    history = str(params.get("history", "")).lower()
    if history in ("highly_suspicious", "highly suspicious", "2"):
        score += 2; breakdown.append("History highly suspicious: +2")
    elif history in ("moderately_suspicious", "moderately suspicious", "1"):
        score += 1; breakdown.append("History moderately suspicious: +1")
    else:
        breakdown.append("History slightly suspicious: +0")
    # ECG
    ecg = str(params.get("ecg", "")).lower()
    if ecg in ("significant_st_deviation", "significant st deviation", "2"):
        score += 2; breakdown.append("ECG significant ST deviation: +2")
    elif ecg in ("non_specific_repolarisation", "lbbb", "lv_hypertrophy", "1"):
        score += 1; breakdown.append("ECG non-specific repolarisation disturbance: +1")
    else:
        breakdown.append("ECG normal: +0")
    # Age
    age = int(params.get("age", 0))
    if age >= 65:
        score += 2; breakdown.append(f"Age ≥65 ({age}): +2")
    elif age >= 45:
        score += 1; breakdown.append(f"Age 45–64 ({age}): +1")
    else:
        breakdown.append(f"Age <45 ({age}): +0")
    # Risk factors (HTN, hypercholesterolaemia, DM, obesity BMI>30, smoking, family hx)
    rf = int(params.get("risk_factors", 0))
    if rf >= 3 or params.get("known_atherosclerosis"):
        score += 2; breakdown.append("≥3 risk factors or known atherosclerosis: +2")
    elif rf in (1, 2):
        score += 1; breakdown.append(f"{rf} risk factor(s): +1")
    else:
        breakdown.append("No known risk factors: +0")
    # Troponin
    troponin = str(params.get("troponin", "")).lower()
    if troponin in (">3x_uln", "3", "2"):
        score += 2; breakdown.append("Troponin >3×ULN: +2")
    elif troponin in ("1-3x_uln", "1-3x uln", "1"):
        score += 1; breakdown.append("Troponin 1–3×ULN: +1")
    else:
        breakdown.append("Troponin ≤ULN: +0")
    if score >= 7:
        risk = "High (7–10): ≥72% MACE risk — admit, early invasive strategy"
    elif score >= 4:
        risk = "Moderate (4–6): 12–50% MACE risk — observe, serial troponins"
    else:
        risk = "Low (0–3): ≤1.7% MACE risk — consider early discharge"
    return {"score": score, "mace_risk": risk, "breakdown": breakdown}


def _score_sofa(params: Dict[str, Any]) -> Dict[str, Any]:
    """Sequential Organ Failure Assessment (SOFA) score for sepsis severity."""
    score = 0
    breakdown = []
    # Respiration: PaO2/FiO2 ratio
    pf = float(params.get("pao2_fio2", 400))
    if pf < 100:
        score += 4; breakdown.append(f"PaO2/FiO2 <100 ({pf:.0f}): +4")
    elif pf < 200:
        score += 3; breakdown.append(f"PaO2/FiO2 <200 ({pf:.0f}): +3")
    elif pf < 300:
        score += 2; breakdown.append(f"PaO2/FiO2 <300 ({pf:.0f}): +2")
    elif pf < 400:
        score += 1; breakdown.append(f"PaO2/FiO2 <400 ({pf:.0f}): +1")
    else:
        breakdown.append(f"PaO2/FiO2 ≥400 ({pf:.0f}): +0")
    # Coagulation: platelets (10³/μL)
    plt = float(params.get("platelets", 200))
    if plt < 20:
        score += 4; breakdown.append(f"Platelets <20 ({plt:.0f}): +4")
    elif plt < 50:
        score += 3; breakdown.append(f"Platelets <50 ({plt:.0f}): +3")
    elif plt < 100:
        score += 2; breakdown.append(f"Platelets <100 ({plt:.0f}): +2")
    elif plt < 150:
        score += 1; breakdown.append(f"Platelets <150 ({plt:.0f}): +1")
    else:
        breakdown.append(f"Platelets ≥150 ({plt:.0f}): +0")
    # Liver: bilirubin (mg/dL)
    bili = float(params.get("bilirubin", 0))
    if bili >= 12.0:
        score += 4; breakdown.append(f"Bilirubin ≥12 mg/dL ({bili}): +4")
    elif bili >= 6.0:
        score += 3; breakdown.append(f"Bilirubin 6–11.9 mg/dL ({bili}): +3")
    elif bili >= 2.0:
        score += 2; breakdown.append(f"Bilirubin 2–5.9 mg/dL ({bili}): +2")
    elif bili >= 1.2:
        score += 1; breakdown.append(f"Bilirubin 1.2–1.9 mg/dL ({bili}): +1")
    else:
        breakdown.append(f"Bilirubin <1.2 mg/dL ({bili}): +0")
    # Cardiovascular: MAP or vasopressors
    map_val = float(params.get("map", 70))
    vasopressor = str(params.get("vasopressor", "none")).lower()
    if vasopressor in ("dopamine_15", "epinephrine", "norepinephrine", "high"):
        score += 4; breakdown.append("Vasopressors (high dose): +4")
    elif vasopressor in ("dopamine_5_15", "moderate"):
        score += 3; breakdown.append("Vasopressors (moderate dose): +3")
    elif vasopressor in ("dopamine_5", "dobutamine", "low"):
        score += 2; breakdown.append("Vasopressors (low dose): +2")
    elif map_val < 70:
        score += 1; breakdown.append(f"MAP <70 mmHg ({map_val}): +1")
    else:
        breakdown.append(f"MAP ≥70 mmHg ({map_val}): +0")
    # CNS: GCS
    gcs = int(params.get("gcs", 15))
    if gcs < 6:
        score += 4; breakdown.append(f"GCS <6 ({gcs}): +4")
    elif gcs < 10:
        score += 3; breakdown.append(f"GCS 6–9 ({gcs}): +3")
    elif gcs < 13:
        score += 2; breakdown.append(f"GCS 10–12 ({gcs}): +2")
    elif gcs < 15:
        score += 1; breakdown.append(f"GCS 13–14 ({gcs}): +1")
    else:
        breakdown.append(f"GCS 15 ({gcs}): +0")
    # Renal: creatinine (mg/dL) or urine output
    cr = float(params.get("creatinine", 0))
    uo = float(params.get("urine_output_ml_day", 1000))
    if cr >= 5.0 or uo < 200:
        score += 4; breakdown.append(f"Creatinine ≥5 or UO <200 mL/day: +4")
    elif cr >= 3.5 or uo < 500:
        score += 3; breakdown.append(f"Creatinine 3.5–4.9 or UO <500 mL/day: +3")
    elif cr >= 2.0:
        score += 2; breakdown.append(f"Creatinine 2.0–3.4 mg/dL ({cr}): +2")
    elif cr >= 1.2:
        score += 1; breakdown.append(f"Creatinine 1.2–1.9 mg/dL ({cr}): +1")
    else:
        breakdown.append(f"Creatinine <1.2 mg/dL ({cr}): +0")
    if score >= 11:
        mortality = "Mortality >50%"
    elif score >= 7:
        mortality = "Mortality 15–20%"
    elif score >= 3:
        mortality = "Mortality <10%"
    else:
        mortality = "Mortality <1%"
    return {"score": score, "mortality_estimate": mortality, "breakdown": breakdown}


def _score_qsofa(params: Dict[str, Any]) -> Dict[str, Any]:
    """qSOFA (quick SOFA) for sepsis screening outside ICU."""
    score = 0
    breakdown = []
    rr = int(params.get("respiratory_rate", 0))
    if rr >= 22:
        score += 1; breakdown.append(f"Respiratory rate ≥22 ({rr}): +1")
    sbp = int(params.get("sbp", 120))
    if sbp <= 100:
        score += 1; breakdown.append(f"SBP ≤100 mmHg ({sbp}): +1")
    if params.get("altered_mentation"):
        score += 1; breakdown.append("Altered mentation (GCS <15): +1")
    if score >= 2:
        risk = "High risk: ≥2 criteria — suspect sepsis, further assessment needed"
    else:
        risk = "Low risk: <2 criteria — sepsis less likely but not excluded"
    return {"score": score, "sepsis_risk": risk, "breakdown": breakdown}


def _score_meld(params: Dict[str, Any]) -> Dict[str, Any]:
    """MELD score (Model for End-Stage Liver Disease) for liver transplant prioritisation."""
    inr = float(params.get("inr", 1.0))
    bili = float(params.get("bilirubin", 1.0))  # mg/dL
    cr = float(params.get("creatinine", 1.0))   # mg/dL
    sodium = float(params.get("sodium", 137))    # mEq/L; for MELD-Na
    dialysis = bool(params.get("dialysis"))
    # Values floored at 1.0 per UNOS policy
    inr = max(inr, 1.0)
    bili = max(bili, 1.0)
    cr = max(cr, 1.0)
    if dialysis or cr > 4.0:
        cr = 4.0
    meld = round(10 * (0.957 * math.log(cr) + 0.378 * math.log(bili) + 1.120 * math.log(inr)) + 6.43)
    meld = max(6, min(meld, 40))
    # MELD-Na (UNOS 2016)
    sodium = max(125, min(sodium, 137))
    meld_na = meld - sodium - (0.025 * meld * (140 - sodium)) + 140
    meld_na = round(max(6, min(meld_na, 40)))
    if meld >= 30:
        mortality_90d = ">50% 90-day mortality without transplant"
    elif meld >= 20:
        mortality_90d = "20–50% 90-day mortality without transplant"
    elif meld >= 10:
        mortality_90d = "6–20% 90-day mortality without transplant"
    else:
        mortality_90d = "<6% 90-day mortality without transplant"
    return {
        "meld": meld,
        "meld_na": meld_na,
        "mortality_estimate": mortality_90d,
        "breakdown": [
            f"INR: {inr:.2f}",
            f"Bilirubin: {bili:.2f} mg/dL",
            f"Creatinine: {cr:.2f} mg/dL (capped at 4.0)",
            f"Sodium: {sodium:.0f} mEq/L (for MELD-Na)",
        ],
    }


def _score_abcd2(params: Dict[str, Any]) -> Dict[str, Any]:
    """ABCD² score for 2-day stroke risk after TIA."""
    score = 0
    breakdown = []
    age = int(params.get("age", 0))
    if age >= 60:
        score += 1; breakdown.append(f"Age ≥60 ({age}): +1")
    sbp = int(params.get("sbp", 0))
    dbp = int(params.get("dbp", 0))
    if sbp >= 140 or dbp >= 90:
        score += 1; breakdown.append(f"BP ≥140/90 ({sbp}/{dbp}): +1")
    # Clinical features
    clinical = str(params.get("clinical_feature", "")).lower()
    if clinical in ("unilateral_weakness", "unilateral weakness"):
        score += 2; breakdown.append("Unilateral weakness: +2")
    elif clinical in ("speech_disturbance", "speech disturbance", "dysphasia"):
        score += 1; breakdown.append("Speech disturbance without weakness: +1")
    else:
        breakdown.append("Other symptom: +0")
    # Duration
    duration = int(params.get("duration_minutes", 0))
    if duration >= 60:
        score += 2; breakdown.append(f"Duration ≥60 min ({duration}): +2")
    elif duration >= 10:
        score += 1; breakdown.append(f"Duration 10–59 min ({duration}): +1")
    else:
        breakdown.append(f"Duration <10 min ({duration}): +0")
    if params.get("diabetes"):
        score += 1; breakdown.append("Diabetes: +1")
    if score >= 6:
        risk = "High (6–7): 8.1% 2-day stroke risk"
    elif score >= 4:
        risk = "Moderate (4–5): 4.1% 2-day stroke risk"
    else:
        risk = "Low (0–3): 1.0% 2-day stroke risk"
    return {"score": score, "stroke_risk_2d": risk, "breakdown": breakdown}


def _score_perc(params: Dict[str, Any]) -> Dict[str, Any]:
    """PERC rule for PE exclusion (all criteria must be absent to rule out PE)."""
    criteria = [
        ("age_over_50", "Age >50"),
        ("heart_rate_over_100", "Heart rate >100 bpm"),
        ("spo2_under_95", "SpO2 <95%"),
        ("unilateral_leg_swelling", "Unilateral leg swelling"),
        ("hemoptysis", "Haemoptysis"),
        ("recent_surgery_trauma", "Recent surgery or trauma within 4 weeks"),
        ("prior_dvt_pe", "Prior DVT or PE"),
        ("hormone_use", "Hormone use (OCP, HRT, oestrogen)"),
    ]
    positive = []
    negative = []
    for key, label in criteria:
        if params.get(key):
            positive.append(f"{label}: PRESENT")
        else:
            negative.append(f"{label}: absent")
    if not positive:
        result = "PERC negative: all 8 criteria absent — PE can be excluded in low pre-test probability patients"
    else:
        result = f"PERC positive: {len(positive)} criterion/criteria present — cannot exclude PE, further workup needed"
    return {
        "perc_positive": bool(positive),
        "criteria_present": positive,
        "criteria_absent": negative,
        "interpretation": result,
    }


def _score_news2(params: Dict[str, Any]) -> Dict[str, Any]:
    """NEWS2 (National Early Warning Score 2) for acute illness severity."""
    score = 0
    breakdown = []
    # Respiration rate
    rr = int(params.get("respiratory_rate", 0))
    if rr <= 8:
        score += 3; breakdown.append(f"RR ≤8 ({rr}): +3")
    elif rr <= 11:
        breakdown.append(f"RR 9–11 ({rr}): +0")
    elif rr <= 20:
        breakdown.append(f"RR 12–20 ({rr}): +0")
    elif rr <= 24:
        score += 2; breakdown.append(f"RR 21–24 ({rr}): +2")
    else:
        score += 3; breakdown.append(f"RR ≥25 ({rr}): +3")
    # SpO2 Scale 1 (no hypercapnic respiratory failure)
    spo2 = int(params.get("spo2", 100))
    hypercapnic = bool(params.get("hypercapnic_respiratory_failure"))
    if hypercapnic:
        # Scale 2
        if spo2 <= 83:
            score += 3; breakdown.append(f"SpO2 ≤83% (Scale 2): +3")
        elif spo2 <= 85:
            score += 2; breakdown.append(f"SpO2 84–85% (Scale 2): +2")
        elif spo2 <= 87:
            score += 1; breakdown.append(f"SpO2 86–87% (Scale 2): +1")
        elif spo2 <= 92:
            breakdown.append(f"SpO2 88–92% (Scale 2): +0")
        elif spo2 <= 94:
            score += 1; breakdown.append(f"SpO2 93–94% (Scale 2, on O2): +1")
        elif spo2 <= 96:
            score += 2; breakdown.append(f"SpO2 95–96% (Scale 2, on O2): +2")
        else:
            score += 3; breakdown.append(f"SpO2 ≥97% (Scale 2, on O2): +3")
    else:
        # Scale 1
        if spo2 <= 91:
            score += 3; breakdown.append(f"SpO2 ≤91%: +3")
        elif spo2 <= 93:
            score += 2; breakdown.append(f"SpO2 92–93%: +2")
        elif spo2 <= 95:
            score += 1; breakdown.append(f"SpO2 94–95%: +1")
        else:
            breakdown.append(f"SpO2 ≥96%: +0")
    # Supplemental oxygen
    if params.get("on_oxygen"):
        score += 2; breakdown.append("On supplemental oxygen: +2")
    # Systolic BP
    sbp = int(params.get("sbp", 120))
    if sbp <= 90:
        score += 3; breakdown.append(f"SBP ≤90 ({sbp}): +3")
    elif sbp <= 100:
        score += 2; breakdown.append(f"SBP 91–100 ({sbp}): +2")
    elif sbp <= 110:
        score += 1; breakdown.append(f"SBP 101–110 ({sbp}): +1")
    elif sbp <= 219:
        breakdown.append(f"SBP 111–219 ({sbp}): +0")
    else:
        score += 3; breakdown.append(f"SBP ≥220 ({sbp}): +3")
    # Pulse
    hr = int(params.get("heart_rate", 80))
    if hr <= 40:
        score += 3; breakdown.append(f"HR ≤40 ({hr}): +3")
    elif hr <= 50:
        score += 1; breakdown.append(f"HR 41–50 ({hr}): +1")
    elif hr <= 90:
        breakdown.append(f"HR 51–90 ({hr}): +0")
    elif hr <= 110:
        score += 1; breakdown.append(f"HR 91–110 ({hr}): +1")
    elif hr <= 130:
        score += 2; breakdown.append(f"HR 111–130 ({hr}): +2")
    else:
        score += 3; breakdown.append(f"HR ≥131 ({hr}): +3")
    # Consciousness (AVPU)
    avpu = str(params.get("consciousness", "alert")).lower()
    if avpu in ("alert", "a"):
        breakdown.append("Alert: +0")
    elif avpu in ("new_confusion", "new confusion", "c"):
        score += 3; breakdown.append("New confusion: +3")
    elif avpu in ("voice", "v"):
        score += 3; breakdown.append("Responds to voice: +3")
    elif avpu in ("pain", "p"):
        score += 3; breakdown.append("Responds to pain: +3")
    elif avpu in ("unresponsive", "u"):
        score += 3; breakdown.append("Unresponsive: +3")
    # Temperature
    temp = float(params.get("temperature", 37.0))
    if temp <= 35.0:
        score += 3; breakdown.append(f"Temp ≤35.0°C ({temp}): +3")
    elif temp <= 36.0:
        score += 1; breakdown.append(f"Temp 35.1–36.0°C ({temp}): +1")
    elif temp <= 38.0:
        breakdown.append(f"Temp 36.1–38.0°C ({temp}): +0")
    elif temp <= 39.0:
        score += 1; breakdown.append(f"Temp 38.1–39.0°C ({temp}): +1")
    else:
        score += 2; breakdown.append(f"Temp >39.0°C ({temp}): +2")
    if score >= 7:
        level = "Level 3 (score ≥7): Emergency response — continuous monitoring, critical care review"
    elif score >= 5:
        level = "Level 2 (score 5–6): Urgent response — urgent medical review"
    elif score == 4 or (score >= 1 and any("3" in b for b in breakdown)):
        level = "Level 2 (score 4 or single parameter score 3): Urgent response"
    elif score >= 1:
        level = "Level 1 (score 1–4): Low-medium risk — ward-based monitoring"
    else:
        level = "Level 0 (score 0): Low risk — routine monitoring"
    return {"score": score, "response_level": level, "breakdown": breakdown}


def _score_glasgow_blatchford(params: Dict[str, Any]) -> Dict[str, Any]:
    """Glasgow-Blatchford Score (GBS) for upper GI bleeding risk."""
    score = 0
    breakdown = []
    # BUN (mg/dL)
    bun = float(params.get("bun", 0))
    if bun >= 150:
        score += 6; breakdown.append(f"BUN ≥150 mg/dL ({bun}): +6")
    elif bun >= 100:
        score += 5; breakdown.append(f"BUN 100–149 mg/dL ({bun}): +5")
    elif bun >= 70:
        score += 4; breakdown.append(f"BUN 70–99 mg/dL ({bun}): +4")
    elif bun >= 40:
        score += 3; breakdown.append(f"BUN 40–69 mg/dL ({bun}): +3")
    elif bun >= 23:
        score += 2; breakdown.append(f"BUN 23–39 mg/dL ({bun}): +2")
    elif bun >= 18.2:
        score += 1; breakdown.append(f"BUN 18.2–22.9 mg/dL ({bun}): +1")
    else:
        breakdown.append(f"BUN <18.2 mg/dL ({bun}): +0")
    # Haemoglobin
    sex = str(params.get("sex", "male")).lower()
    hgb = float(params.get("hemoglobin", 0))
    if sex in ("m", "male"):
        if hgb < 10:
            score += 6; breakdown.append(f"Hgb <10 g/dL (male, {hgb}): +6")
        elif hgb < 12:
            score += 3; breakdown.append(f"Hgb 10–11.9 g/dL (male, {hgb}): +3")
        elif hgb < 13:
            score += 1; breakdown.append(f"Hgb 12–12.9 g/dL (male, {hgb}): +1")
        else:
            breakdown.append(f"Hgb ≥13 g/dL (male, {hgb}): +0")
    else:
        if hgb < 10:
            score += 6; breakdown.append(f"Hgb <10 g/dL (female, {hgb}): +6")
        elif hgb < 12:
            score += 1; breakdown.append(f"Hgb 10–11.9 g/dL (female, {hgb}): +1")
        else:
            breakdown.append(f"Hgb ≥12 g/dL (female, {hgb}): +0")
    # SBP
    sbp = int(params.get("sbp", 120))
    if sbp < 90:
        score += 3; breakdown.append(f"SBP <90 ({sbp}): +3")
    elif sbp < 100:
        score += 2; breakdown.append(f"SBP 90–99 ({sbp}): +2")
    elif sbp < 110:
        score += 1; breakdown.append(f"SBP 100–109 ({sbp}): +1")
    else:
        breakdown.append(f"SBP ≥110 ({sbp}): +0")
    # Other
    if params.get("heart_rate_over_100"):
        score += 1; breakdown.append("Heart rate ≥100 bpm: +1")
    if params.get("melena"):
        score += 1; breakdown.append("Melena: +1")
    if params.get("syncope"):
        score += 2; breakdown.append("Syncope: +2")
    if params.get("liver_disease"):
        score += 2; breakdown.append("Liver disease: +2")
    if params.get("cardiac_failure"):
        score += 2; breakdown.append("Cardiac failure: +2")
    if score == 0:
        risk = "Score 0: Very low risk — consider outpatient management"
    elif score <= 2:
        risk = f"Score {score}: Low risk — low likelihood of needing intervention"
    elif score <= 5:
        risk = f"Score {score}: Moderate risk — likely needs intervention"
    else:
        risk = f"Score {score}: High risk — high likelihood of needing intervention"
    return {"score": score, "risk": risk, "breakdown": breakdown}


def _score_rcri(params: Dict[str, Any]) -> Dict[str, Any]:
    """Revised Cardiac Risk Index (RCRI) for perioperative cardiac risk."""
    score = 0
    breakdown = []
    criteria = [
        ("high_risk_surgery", "High-risk surgery (intraperitoneal, intrathoracic, suprainguinal vascular): +1"),
        ("ischemic_heart_disease", "Ischemic heart disease (hx MI, positive stress test, angina, nitrate use, Q waves): +1"),
        ("heart_failure", "Congestive heart failure (hx pulmonary oedema, PND, bilateral rales, S3, CXR redistribution): +1"),
        ("cerebrovascular_disease", "Cerebrovascular disease (hx TIA or stroke): +1"),
        ("insulin_dependent_diabetes", "Insulin-dependent diabetes mellitus: +1"),
        ("creatinine_over_2", "Preoperative creatinine >2.0 mg/dL: +1"),
    ]
    for key, label in criteria:
        if params.get(key):
            score += 1; breakdown.append(label)
    risk_map = {
        0: "Class I: 0.4% risk of major cardiac event",
        1: "Class II: 0.9% risk of major cardiac event",
        2: "Class III: 6.6% risk of major cardiac event",
    }
    risk = risk_map.get(score, "Class IV (≥3): ≥11% risk of major cardiac event")
    return {"score": score, "cardiac_risk": risk, "breakdown": breakdown}


def _score_cockcroft_gault(params: Dict[str, Any]) -> Dict[str, Any]:
    """Cockcroft-Gault creatinine clearance (CrCl) for drug dosing."""
    age = int(params.get("age", 0))
    weight_kg = float(params.get("weight_kg", 0))
    scr = float(params.get("creatinine_mgdl", 0))
    sex = str(params.get("sex", "male")).lower()
    if age <= 0 or weight_kg <= 0 or scr <= 0:
        raise ValueError("'age', 'weight_kg', and 'creatinine_mgdl' are required.")
    # Use ideal body weight if actual > IBW (optional; use actual by default)
    ibw = 50 + 2.3 * (float(params.get("height_cm", 0)) - 152.4) / 2.54 if params.get("height_cm") else None
    if ibw and sex not in ("f", "female"):
        effective_weight = min(weight_kg, ibw) if ibw > 0 else weight_kg
    elif ibw and sex in ("f", "female"):
        ibw_f = 45.5 + 2.3 * (float(params.get("height_cm", 0)) - 152.4) / 2.54
        effective_weight = min(weight_kg, ibw_f) if ibw_f > 0 else weight_kg
    else:
        effective_weight = weight_kg
    crcl = (140 - age) * effective_weight / (72 * scr)
    if sex in ("f", "female"):
        crcl *= 0.85
    crcl = round(crcl, 1)
    if crcl >= 90:
        stage = "Normal (≥90 mL/min)"
    elif crcl >= 60:
        stage = "Mild reduction (60–89 mL/min)"
    elif crcl >= 30:
        stage = "Moderate reduction (30–59 mL/min) — dose adjust many drugs"
    elif crcl >= 15:
        stage = "Severe reduction (15–29 mL/min) — significant dose adjustment required"
    else:
        stage = "Kidney failure (<15 mL/min) — consider renal replacement therapy"
    return {
        "crcl_ml_min": crcl,
        "stage": stage,
        "effective_weight_kg": round(effective_weight, 1),
        "breakdown": [
            f"Age: {age} yr, Weight: {weight_kg} kg (effective: {effective_weight:.1f} kg)",
            f"Creatinine: {scr} mg/dL, Sex: {sex}",
            "Formula: (140−age) × weight / (72 × SCr)" + (
                " × 0.85 (female)" if sex in ("f", "female") else ""
            ),
        ],
    }


def _score_mdrd(params: Dict[str, Any]) -> Dict[str, Any]:
    """MDRD eGFR (4-variable MDRD Study Equation)."""
    scr = float(params.get("creatinine_mgdl", 0))
    age = int(params.get("age", 0))
    sex = str(params.get("sex", "male")).lower()
    if scr <= 0 or age <= 0:
        raise ValueError("'creatinine_mgdl' and 'age' are required.")
    egfr = 175 * (scr ** -1.154) * (age ** -0.203)
    if sex in ("f", "female"):
        egfr *= 0.742
    egfr = round(egfr, 1)
    if egfr >= 90:
        stage = "G1 (Normal or high, ≥90)"
    elif egfr >= 60:
        stage = "G2 (Mildly decreased, 60–89)"
    elif egfr >= 45:
        stage = "G3a (45–59)"
    elif egfr >= 30:
        stage = "G3b (30–44)"
    elif egfr >= 15:
        stage = "G4 (15–29)"
    else:
        stage = "G5 (Kidney failure, <15)"
    return {
        "egfr_ml_min_1.73m2": egfr,
        "ckd_stage": stage,
        "note": "MDRD 4-variable equation (race-free). CKD-EPI 2021 preferred for GFR >60.",
    }


_SCORE_REGISTRY: Dict[str, Any] = {
    "cha2ds2_vasc": _score_cha2ds2_vasc,
    "cha2ds2-vasc": _score_cha2ds2_vasc,
    "wells_dvt": _score_wells_dvt,
    "wells_pe": _score_wells_pe,
    "curb65": _score_curb65,
    "curb-65": _score_curb65,
    "child_pugh": _score_child_pugh,
    "child-pugh": _score_child_pugh,
    "gcs": _score_gcs,
    "glasgow_coma_scale": _score_gcs,
    "bmi": _score_bmi,
    "egfr": _score_egfr_ckd_epi,
    "egfr_ckd_epi": _score_egfr_ckd_epi,
    # New scores
    "has_bled": _score_has_bled,
    "has-bled": _score_has_bled,
    "hasbled": _score_has_bled,
    "heart": _score_heart,
    "heart_score": _score_heart,
    "sofa": _score_sofa,
    "qsofa": _score_qsofa,
    "q_sofa": _score_qsofa,
    "meld": _score_meld,
    "meld_na": _score_meld,
    "abcd2": _score_abcd2,
    "perc": _score_perc,
    "perc_rule": _score_perc,
    "news2": _score_news2,
    "news": _score_news2,
    "glasgow_blatchford": _score_glasgow_blatchford,
    "gbs": _score_glasgow_blatchford,
    "rcri": _score_rcri,
    "revised_cardiac_risk_index": _score_rcri,
    "cockcroft_gault": _score_cockcroft_gault,
    "crcl": _score_cockcroft_gault,
    "mdrd": _score_mdrd,
    "mdrd_egfr": _score_mdrd,
}

_SCORE_DESCRIPTIONS = {
    "cha2ds2_vasc": "CHA₂DS₂-VASc stroke risk in atrial fibrillation",
    "wells_dvt": "Wells DVT probability score",
    "wells_pe": "Wells PE probability score",
    "curb65": "CURB-65 pneumonia severity",
    "child_pugh": "Child-Pugh liver cirrhosis severity",
    "gcs": "Glasgow Coma Scale",
    "bmi": "Body Mass Index",
    "egfr": "eGFR (CKD-EPI 2021 creatinine equation)",
    # New scores
    "has_bled": "HAS-BLED bleeding risk score for anticoagulated AF patients",
    "heart": "HEART score for MACE risk in chest pain (History/ECG/Age/Risk factors/Troponin)",
    "sofa": "SOFA (Sequential Organ Failure Assessment) score for sepsis severity",
    "qsofa": "qSOFA quick sepsis screening score (RR, SBP, mentation)",
    "meld": "MELD score (Model for End-Stage Liver Disease) for transplant prioritisation",
    "abcd2": "ABCD² score for 2-day stroke risk after TIA",
    "perc": "PERC rule for pulmonary embolism exclusion (8 criteria)",
    "news2": "NEWS2 (National Early Warning Score 2) for acute illness severity",
    "glasgow_blatchford": "Glasgow-Blatchford Score for upper GI bleeding risk",
    "rcri": "Revised Cardiac Risk Index (RCRI) for perioperative cardiac risk",
    "cockcroft_gault": "Cockcroft-Gault creatinine clearance (CrCl) for drug dosing",
    "mdrd": "MDRD 4-variable eGFR equation",
}


@dataclass
class ClinicalScoreCalculatorTool:
    """Compute widely-used clinical risk scores and indices.

    Supported scores:
      - ``cha2ds2_vasc``       : CHA₂DS₂-VASc stroke risk in atrial fibrillation
      - ``has_bled``           : HAS-BLED bleeding risk (AF anticoagulation)
      - ``heart``              : HEART score for chest pain MACE risk
      - ``wells_dvt``          : Wells DVT probability
      - ``wells_pe``           : Wells PE probability
      - ``perc``               : PERC rule for PE exclusion
      - ``curb65``             : CURB-65 pneumonia severity
      - ``sofa``               : SOFA sepsis severity score
      - ``qsofa``              : qSOFA sepsis screening
      - ``news2``              : NEWS2 early warning score
      - ``child_pugh``         : Child-Pugh liver cirrhosis class
      - ``meld``               : MELD / MELD-Na liver transplant score
      - ``gcs``                : Glasgow Coma Scale
      - ``abcd2``              : ABCD² TIA stroke risk
      - ``glasgow_blatchford`` : Glasgow-Blatchford upper GI bleeding score
      - ``rcri``               : Revised Cardiac Risk Index (perioperative)
      - ``bmi``                : Body Mass Index
      - ``egfr``               : eGFR via CKD-EPI 2021 (creatinine-based)
      - ``cockcroft_gault``    : Cockcroft-Gault CrCl for drug dosing
      - ``mdrd``               : MDRD 4-variable eGFR

    Args:
        name: Tool name used in agent prompts and the tool registry.
        description: Human-readable description injected into the system prompt.
    """

    name: str = "clinical_score_calculator"
    description: str = (
        "Calculate clinical risk scores: CHA₂DS₂-VASc, HAS-BLED, HEART, Wells DVT/PE, PERC, "
        "CURB-65, SOFA/qSOFA, NEWS2, Child-Pugh, MELD, GCS, ABCD², Glasgow-Blatchford, RCRI, "
        "BMI, eGFR, Cockcroft-Gault, MDRD. Pass score_name='list_scores' to see all parameters."
    )

    def run(self, payload: Dict[str, Any]) -> ToolResult:
        """Calculate a clinical score.

        Args:
            payload: Must contain ``"score_name"`` (str): the score to calculate.
                Pass ``"score_name": "list_scores"`` to list all available scores
                and their parameters. All other fields are score-specific parameters.

        Returns:
            ToolResult with the calculated score, interpretation, and breakdown.

        Raises:
            ValueError: If ``score_name`` is missing, unsupported, or required
                parameters are missing.
        """
        score_name = str(payload.get("score_name", "")).strip().lower()
        if not score_name:
            raise ValueError(
                "'score_name' is required. "
                "Pass 'list_scores' to see all available scores."
            )

        if score_name == "list_scores":
            return ToolResult(
                name=self.name,
                payload={
                    "available_scores": {
                        k: v for k, v in _SCORE_DESCRIPTIONS.items()
                    },
                    "note": (
                        "Call with score_name and the required parameters. "
                        "Example: {\"score_name\": \"cha2ds2_vasc\", \"age\": 72, "
                        "\"hypertension\": true, \"diabetes\": false, \"sex\": \"male\"}"
                    ),
                },
            )

        # Normalise name
        normalised = score_name.replace("-", "_").replace(" ", "_")
        fn = _SCORE_REGISTRY.get(normalised) or _SCORE_REGISTRY.get(score_name)
        if fn is None:
            available = sorted(set(_SCORE_DESCRIPTIONS.keys()))
            raise ValueError(
                f"Unknown score: '{score_name}'. "
                f"Available scores: {available}"
            )

        try:
            result = fn(payload)
        except (KeyError, TypeError) as e:
            raise ValueError(
                f"Missing or invalid parameter for '{score_name}': {e}. "
                "Pass 'list_scores' to see required parameters."
            )

        result["score_name"] = score_name
        return ToolResult(name=self.name, payload=result)

    def get_params_schema(self):
        """Return structured parameter schema for prompt injection."""
        return [
            # ----------------------------------------------------------------
            # Selector
            # ----------------------------------------------------------------
            ToolParam(
                name="score_name",
                type="string",
                required=True,
                description=(
                    "Name of the clinical score to calculate. Supported values: "
                    "\"cha2ds2_vasc\" (AF stroke risk), "
                    "\"has_bled\" (AF anticoagulation bleeding risk), "
                    "\"heart\" (chest pain MACE risk), "
                    "\"wells_dvt\" (DVT probability), "
                    "\"wells_pe\" (PE probability), "
                    "\"perc\" (PE exclusion rule), "
                    "\"curb65\" (pneumonia severity), "
                    "\"sofa\" (sepsis organ failure), "
                    "\"qsofa\" (quick sepsis screening), "
                    "\"news2\" (National Early Warning Score 2), "
                    "\"child_pugh\" (liver cirrhosis class), "
                    "\"meld\" (end-stage liver disease / transplant), "
                    "\"gcs\" (Glasgow Coma Scale), "
                    "\"abcd2\" (TIA stroke risk), "
                    "\"glasgow_blatchford\" (upper GI bleeding risk), "
                    "\"rcri\" (perioperative cardiac risk), "
                    "\"bmi\" (Body Mass Index), "
                    "\"egfr\" (eGFR via CKD-EPI 2021), "
                    "\"cockcroft_gault\" (creatinine clearance for drug dosing), "
                    "\"mdrd\" (MDRD 4-variable eGFR). "
                    "Pass \"list_scores\" to retrieve the full parameter list for every score."
                ),
            ),
            # ----------------------------------------------------------------
            # Demographics (shared across many scores)
            # ----------------------------------------------------------------
            ToolParam(
                name="age",
                type="integer",
                required=False,
                description=(
                    "Patient age in years. "
                    "Required for: cha2ds2_vasc, has_bled, curb65, egfr, "
                    "abcd2, cockcroft_gault, mdrd."
                ),
            ),
            ToolParam(
                name="sex",
                type="string",
                required=False,
                description=(
                    "Patient biological sex: \"male\" or \"female\". "
                    "Required for: cha2ds2_vasc, egfr, cockcroft_gault, mdrd, "
                    "glasgow_blatchford."
                ),
            ),
            ToolParam(
                name="weight_kg",
                type="number",
                required=False,
                description=(
                    "Patient weight in kilograms. "
                    "Required for: bmi, cockcroft_gault."
                ),
            ),
            ToolParam(
                name="height_m",
                type="number",
                required=False,
                description="Patient height in metres. Required for: bmi. (height_cm also accepted)",
            ),
            ToolParam(
                name="height_cm",
                type="number",
                required=False,
                description="Patient height in centimetres. Alternative to height_m for bmi.",
            ),
            # ----------------------------------------------------------------
            # Renal / lab values
            # ----------------------------------------------------------------
            ToolParam(
                name="creatinine_mgdl",
                type="number",
                required=False,
                description=(
                    "Serum creatinine in mg/dL. "
                    "Required for: egfr, cockcroft_gault, mdrd, sofa."
                ),
            ),
            ToolParam(
                name="bun_mgdl",
                type="number",
                required=False,
                description=(
                    "Blood urea nitrogen in mg/dL (>19 mg/dL = 1 pt). "
                    "Used in: curb65."
                ),
            ),
            ToolParam(
                name="bun",
                type="number",
                required=False,
                description=(
                    "Blood urea nitrogen in mg/dL for glasgow_blatchford scoring "
                    "(thresholds: 18.2–22.4, 22.4–28, 28–70, ≥70 mg/dL)."
                ),
            ),
            ToolParam(
                name="inr",
                type="number",
                required=False,
                description="INR (international normalised ratio). Required for: meld.",
            ),
            ToolParam(
                name="bilirubin",
                type="number",
                required=False,
                description=(
                    "Total bilirubin in mg/dL. "
                    "Required for: meld, sofa. "
                    "Used in: child_pugh (as bilirubin_mgdl)."
                ),
            ),
            ToolParam(
                name="sodium",
                type="number",
                required=False,
                description="Serum sodium in mmol/L. Optional for: meld (enables MELD-Na).",
            ),
            ToolParam(
                name="platelets",
                type="number",
                required=False,
                description="Platelet count ×10⁹/L. Required for: sofa.",
            ),
            ToolParam(
                name="pao2_fio2",
                type="number",
                required=False,
                description=(
                    "PaO₂/FiO₂ ratio (mmHg). "
                    "Required for: sofa (respiratory component)."
                ),
            ),
            ToolParam(
                name="hemoglobin",
                type="number",
                required=False,
                description=(
                    "Haemoglobin in g/dL. "
                    "Required for: glasgow_blatchford."
                ),
            ),
            # ----------------------------------------------------------------
            # Vital signs
            # ----------------------------------------------------------------
            ToolParam(
                name="sbp",
                type="integer",
                required=False,
                description=(
                    "Systolic blood pressure in mmHg. "
                    "Used in: qsofa (<100 = 1 pt), abcd2 (≥140 = 1 pt), "
                    "news2, glasgow_blatchford."
                ),
            ),
            ToolParam(
                name="systolic_bp",
                type="integer",
                required=False,
                description=(
                    "Systolic blood pressure in mmHg (<90 = 1 pt). "
                    "Used in: curb65. (Alias for sbp in other scores.)"
                ),
            ),
            ToolParam(
                name="respiratory_rate",
                type="integer",
                required=False,
                description=(
                    "Respiratory rate in breaths/min. "
                    "Used in: curb65 (≥30 = 1 pt), wells_pe, qsofa (≥22 = 1 pt), news2."
                ),
            ),
            ToolParam(
                name="heart_rate",
                type="integer",
                required=False,
                description="Heart rate in beats/min. Used in: news2.",
            ),
            ToolParam(
                name="temperature",
                type="number",
                required=False,
                description="Body temperature in °C. Used in: news2.",
            ),
            ToolParam(
                name="spo2",
                type="number",
                required=False,
                description="Oxygen saturation (SpO₂) as a percentage (e.g., 96). Used in: news2.",
            ),
            # ----------------------------------------------------------------
            # Consciousness / neurological
            # ----------------------------------------------------------------
            ToolParam(
                name="gcs",
                type="integer",
                required=False,
                description=(
                    "Glasgow Coma Scale total (3–15). "
                    "Used in: sofa (GCS <15 scores points)."
                ),
            ),
            ToolParam(
                name="consciousness",
                type="string",
                required=False,
                description=(
                    "Level of consciousness for NEWS2. "
                    "One of: \"alert\", \"voice\", \"pain\", \"unresponsive\" "
                    "(AVPU scale; any non-alert = 3 pts)."
                ),
            ),
            ToolParam(
                name="altered_mentation",
                type="boolean",
                required=False,
                description="Altered mental status (GCS <15 or new confusion). Used in: qsofa (1 pt).",
            ),
            ToolParam(
                name="confusion",
                type="boolean",
                required=False,
                description="New-onset confusion or disorientation. Used in: curb65 (1 pt).",
            ),
            ToolParam(
                name="eye_opening",
                type="integer",
                required=False,
                description="GCS eye opening sub-score (1–4). Required for: gcs.",
            ),
            ToolParam(
                name="verbal_response",
                type="integer",
                required=False,
                description="GCS verbal response sub-score (1–5). Required for: gcs.",
            ),
            ToolParam(
                name="motor_response",
                type="integer",
                required=False,
                description="GCS motor response sub-score (1–6). Required for: gcs.",
            ),
            # ----------------------------------------------------------------
            # Cardiac / AF / bleeding (cha2ds2_vasc, has_bled, heart, rcri)
            # ----------------------------------------------------------------
            ToolParam(
                name="hypertension",
                type="boolean",
                required=False,
                description=(
                    "History of hypertension. "
                    "Used in: cha2ds2_vasc, has_bled (uncontrolled SBP >160 = 1 pt)."
                ),
            ),
            ToolParam(
                name="diabetes",
                type="boolean",
                required=False,
                description="History of diabetes mellitus. Used in: cha2ds2_vasc, abcd2.",
            ),
            ToolParam(
                name="stroke",
                type="boolean",
                required=False,
                description="Prior stroke or TIA. Used in: cha2ds2_vasc (2 pts).",
            ),
            ToolParam(
                name="chf",
                type="boolean",
                required=False,
                description="Congestive heart failure / LV dysfunction. Used in: cha2ds2_vasc.",
            ),
            ToolParam(
                name="vascular_disease",
                type="boolean",
                required=False,
                description=(
                    "Vascular disease (prior MI, peripheral artery disease, aortic plaque). "
                    "Used in: cha2ds2_vasc."
                ),
            ),
            ToolParam(
                name="labile_inr",
                type="boolean",
                required=False,
                description="Labile INR (TTR <60%). Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="renal_disease",
                type="boolean",
                required=False,
                description="Renal disease (dialysis, transplant, Cr >2.26 mg/dL). Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="liver_disease",
                type="boolean",
                required=False,
                description="Liver disease (cirrhosis, bilirubin >2×ULN + AST/ALT/ALP >3×ULN). Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="bleeding_history",
                type="boolean",
                required=False,
                description="Prior bleeding history or predisposition. Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="antiplatelet_nsaid",
                type="boolean",
                required=False,
                description="Antiplatelet or NSAID use. Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="alcohol",
                type="boolean",
                required=False,
                description="Alcohol use (≥8 drinks/week). Used in: has_bled (1 pt).",
            ),
            ToolParam(
                name="ischemic_heart_disease",
                type="boolean",
                required=False,
                description="History of ischaemic heart disease. Used in: rcri (1 pt).",
            ),
            ToolParam(
                name="high_risk_surgery",
                type="boolean",
                required=False,
                description=(
                    "High-risk surgery (intraperitoneal, intrathoracic, or suprainguinal vascular). "
                    "Used in: rcri (1 pt)."
                ),
            ),
            ToolParam(
                name="heart_failure_history",
                type="boolean",
                required=False,
                description="History of heart failure. Used in: rcri (1 pt).",
            ),
            ToolParam(
                name="cerebrovascular_disease",
                type="boolean",
                required=False,
                description="History of cerebrovascular disease (stroke or TIA). Used in: rcri (1 pt).",
            ),
            ToolParam(
                name="insulin_dependent_diabetes",
                type="boolean",
                required=False,
                description="Insulin-dependent diabetes mellitus. Used in: rcri (1 pt).",
            ),
            ToolParam(
                name="preoperative_creatinine_over_2",
                type="boolean",
                required=False,
                description="Pre-operative creatinine >2.0 mg/dL. Used in: rcri (1 pt).",
            ),
            # ----------------------------------------------------------------
            # HEART score parameters
            # ----------------------------------------------------------------
            ToolParam(
                name="history",
                type="string",
                required=False,
                description=(
                    "Clinical history suspicion for ACS. "
                    "One of: \"other\" (0 pts), \"moderately_suspicious\" (1 pt), "
                    "\"highly_suspicious\" (2 pts). "
                    "Required for: heart."
                ),
            ),
            ToolParam(
                name="ecg",
                type="string",
                required=False,
                description=(
                    "ECG findings. One of: \"normal\" (0 pts), \"non_specific\" or \"1\" (1 pt), "
                    "\"significant_st_deviation\" or \"2\" (2 pts). "
                    "Required for: heart."
                ),
            ),
            ToolParam(
                name="risk_factors",
                type="integer",
                required=False,
                description=(
                    "Number of cardiac risk factors (HTN, hypercholesterolaemia, DM, obesity, "
                    "smoking, family history). 0 = 0 pts, 1–2 = 1 pt, ≥3 or known atherosclerosis = 2 pts. "
                    "Used in: heart."
                ),
            ),
            ToolParam(
                name="known_atherosclerosis",
                type="boolean",
                required=False,
                description="Known atherosclerotic disease (overrides risk_factors to 2 pts). Used in: heart.",
            ),
            ToolParam(
                name="troponin",
                type="string",
                required=False,
                description=(
                    "Troponin level relative to ULN. "
                    "One of: \"normal\" (0 pts), \"1-3x_uln\" (1 pt), \">3x_uln\" (2 pts). "
                    "Required for: heart."
                ),
            ),
            # ----------------------------------------------------------------
            # DVT / PE / PERC parameters
            # ----------------------------------------------------------------
            ToolParam(
                name="age_over_50",
                type="boolean",
                required=False,
                description="Age >50 years. Used in: perc (1 criterion).",
            ),
            ToolParam(
                name="heart_rate_over_100",
                type="boolean",
                required=False,
                description="Heart rate >100 bpm. Used in: perc (1 criterion).",
            ),
            ToolParam(
                name="spo2_under_95",
                type="boolean",
                required=False,
                description="SpO₂ <95% on room air. Used in: perc (1 criterion).",
            ),
            ToolParam(
                name="unilateral_leg_swelling",
                type="boolean",
                required=False,
                description="Unilateral leg swelling. Used in: perc, wells_dvt.",
            ),
            ToolParam(
                name="hemoptysis",
                type="boolean",
                required=False,
                description="Haemoptysis. Used in: perc, wells_pe.",
            ),
            ToolParam(
                name="recent_surgery_trauma",
                type="boolean",
                required=False,
                description="Recent surgery or trauma (within 4 weeks). Used in: perc.",
            ),
            ToolParam(
                name="prior_dvt_pe",
                type="boolean",
                required=False,
                description="Prior DVT or PE. Used in: perc, wells_dvt, wells_pe.",
            ),
            ToolParam(
                name="hormone_use",
                type="boolean",
                required=False,
                description="Oestrogen use (OCP, HRT, etc.). Used in: perc.",
            ),
            # ----------------------------------------------------------------
            # TIA (ABCD2) parameters
            # ----------------------------------------------------------------
            ToolParam(
                name="clinical_feature",
                type="string",
                required=False,
                description=(
                    "Focal neurological symptom type for ABCD2. "
                    "One of: \"unilateral_weakness\" (2 pts), "
                    "\"speech_disturbance\" (1 pt), \"other\" (0 pts)."
                ),
            ),
            ToolParam(
                name="duration_minutes",
                type="integer",
                required=False,
                description=(
                    "Duration of TIA symptoms in minutes. "
                    "≥60 min = 2 pts, 10–59 min = 1 pt, <10 min = 0 pts. "
                    "Used in: abcd2."
                ),
            ),
            # ----------------------------------------------------------------
            # GI bleeding (Glasgow-Blatchford) parameters
            # ----------------------------------------------------------------
            ToolParam(
                name="melena",
                type="boolean",
                required=False,
                description="Presence of melaena on presentation. Used in: glasgow_blatchford (1 pt).",
            ),
            ToolParam(
                name="syncope",
                type="boolean",
                required=False,
                description="Syncope on presentation. Used in: glasgow_blatchford (2 pts).",
            ),
            ToolParam(
                name="hepatic_disease",
                type="boolean",
                required=False,
                description="Known hepatic disease. Used in: glasgow_blatchford (2 pts).",
            ),
            ToolParam(
                name="cardiac_failure",
                type="boolean",
                required=False,
                description="Known cardiac failure. Used in: glasgow_blatchford (2 pts).",
            ),
            # ----------------------------------------------------------------
            # Liver (Child-Pugh) parameters
            # ----------------------------------------------------------------
            ToolParam(
                name="bilirubin_mgdl",
                type="number",
                required=False,
                description="Total bilirubin in mg/dL. Used in: child_pugh.",
            ),
            ToolParam(
                name="albumin_gdl",
                type="number",
                required=False,
                description="Serum albumin in g/dL. Used in: child_pugh.",
            ),
            ToolParam(
                name="pt_seconds",
                type="number",
                required=False,
                description="Prothrombin time prolongation in seconds above normal. Used in: child_pugh.",
            ),
            ToolParam(
                name="ascites",
                type="string",
                required=False,
                description=(
                    "Ascites severity. One of: \"none\" (1 pt), \"mild\" (2 pts), "
                    "\"moderate_severe\" (3 pts). Used in: child_pugh."
                ),
            ),
            ToolParam(
                name="encephalopathy",
                type="string",
                required=False,
                description=(
                    "Hepatic encephalopathy grade. One of: \"none\" (1 pt), "
                    "\"grade_1_2\" (2 pts), \"grade_3_4\" (3 pts). Used in: child_pugh."
                ),
            ),
        ]

    def get_usage_example(self) -> str:
        return (
            '{"score_name": "cha2ds2_vasc", "age": 72, "hypertension": true, '
            '"diabetes": false, "stroke": false, "chf": false, '
            '"vascular_disease": false, "sex": "male"}'
        )
