"""
Generator: domain knowledge extraction, seed formula generation, offspring description.

Responsibilities:
- generate_domain_knowledge: Analyze the task domain, output formula priors + preprocessing rules
- initialize_seeds: Generate initial seed formulas based on domain knowledge
- describe / describe_batch: Generate natural language structural descriptions for formulas
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import json_repair
import sympy as sp

from .llm_client import (
    add_guided_json_schema,
    build_openai_client,
    chat_completion,
    resolve_completion_max_tokens,
    with_json_only_instruction,
)


def _extract_code_block(raw: str) -> str:
    """If the response contains a ```json ... ``` code block, extract its content."""
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    return match.group(1) if match else raw


# ------------------------------------------------------------------ #
# Domain Knowledge Prompts
# ------------------------------------------------------------------ #

_DK_PROMPT_HEADER = """\
You are a cross-disciplinary mathematical modeling expert, proficient in classical formulas and mathematical models from physics, chemistry, biology, engineering, economics, and other fields.

Given a background description of a symbolic regression task (including variable names and their physical meanings), please:
1. Analyze the most likely scientific domain/sub-domain for this task
2. List classical formula structures related to these variables in that domain

## Output Format
Return only a JSON object:
{{"domain": "<domain name>", "analysis": "<1-2 sentence domain analysis>", "formulas": [...], {preprocessing_placeholder}"heuristics": [...]}}

- domain: The scientific domain this task belongs to (e.g., "chemical kinetics", "fluid mechanics", "population ecology", etc.)
- analysis: Briefly explain why it belongs to this domain and the possible physical relationships between variables
- formulas: List 5~15 common formula **structural patterns** in this domain, requirements:
  - Use the actual variable names from the task as much as possible
  - Arrange from simple to complex
  - Include function forms specific to this domain (not limited to generic polynomials)
  - Each entry formatted as "Name: formula form (applicable scenario)"
  - Example: "Arrhenius equation: k = c0 * exp(c1 / T) (relationship between chemical reaction rate and temperature)"
  - Example: "Power-law scaling: y = c0 * x**c1 (scale-free scaling relationship)"
  - **Constant notation convention**: Constant parameters must appear in multiplicative form and must not be placed in the denominator.\
    Correct: `c0 * x`, `c0 / (x + c1)`; Incorrect: `x / c0`, `sin(x / c0)`.\
    To express "scaled variable", always write `c * var` instead of `var / c`.
  - The following classical named formulas are pre-built into the system and **should not be repeated**:
    - Arrhenius equation: k = c0 * exp(c1 / T) (chemical reaction rate dependence on temperature)
    - Coulomb's / Newton's inverse-square law: F = c0 / r**2 (gravitational or electrostatic force)
    - Stefan-Boltzmann law: P = c0 * T**4 (blackbody radiation power vs temperature)
    - Beer-Lambert law: I = c0 * exp(c1 * x) (light intensity attenuation through a medium)
    - Michaelis-Menten kinetics: v = c0 * S / (S + c1) (enzyme-catalyzed reaction saturation)
    - Hill equation: y = c0 * x**c1 / (x**c1 + c2) (cooperative binding, sigmoidal dose-response)
    - Kepler's third law: T**2 = c0 * a**3 (orbital period vs semi-major axis)
    - Newton's law of cooling: T = c0 + c1 * exp(c2 * t) (exponential thermal relaxation)
    - Clausius-Clapeyron relation: log(P) = c0 + c1 / T (vapor pressure vs temperature)
    - Langmuir adsorption isotherm: theta = c0 * P / (1 + c1 * P) (surface adsorption equilibrium)
    - Freundlich isotherm: q = c0 * C**c1 (empirical adsorption on heterogeneous surface)
    - Stokes' drag: F = c0 * r * v (viscous drag force on a sphere in fluid)
    - Logistic growth: N = c0 / (1 + c1 * exp(c2 * t)) (population dynamics with carrying capacity)
    - Planck-Wien approximation: I = c0 * f**3 * exp(c1 * f / T) (thermal radiation spectrum)
  Please provide **additional** 5~15 domain-specific formula structures that complement and do not repeat the above pre-built entries."""

_DK_PROMPT_PREPROCESSING = """
- preprocessing: List 1~3 **variable preprocessing suggestions** (at most 3!), each must be one of the following two types:
  - `linear_scale`: Linear scaling x → c1*x + c2, suitable for variables that need offset or scaling
  - `moment`: Moment transform x → x^c1, suitable for variables that need power transformation
  Each entry formatted as a JSON object: `{{"variable": "<variable name>", "type": "linear_scale or moment", "reason": "<brief reason>"}}`
  Examples:
  - `{{"variable": "T", "type": "linear_scale", "reason": "Temperature often needs offset relative to a reference temperature T→c1*T+c2"}}`
  - `{{"variable": "epsilon", "type": "moment", "reason": "Strain may need power transformation ε→ε^c1 to capture nonlinear response"}}`
  Note: Each variable appears at most once; when uncertain, prefer fewer suggestions and only include those with high confidence"""

_DK_PROMPT_FOOTER = """
- heuristics: List 5~10 **domain-specific** formula structure heuristic features describing what physical behavior corresponds to what mathematical structure.
  The following general heuristic features are pre-built into the system and **should not be repeated**:
  - Oscillation damping envelope → Exponential decay envelope f(x)*exp(-c*Abs(x)), Gaussian envelope f(x)*exp(-c*x**2)
  - Saturation/clipping effects → Rational fraction x/(1+x), tanh, selective Abs
  - Growth/decay → exp(c*t), Logistic 1/(1+exp(-c*x)), power law x**c
  - Threshold/activation behavior → Max(x-c, 0), piecewise functions
  - Coupling/cross effects → Variable products x*y, power-law coupling x**c1 * y**c2
  Please provide **additional** 5~10 heuristic features that complement and do not repeat the above pre-built entries.
"""


_DK_DIMENSION_PROMPT_HEADER = """\
You are a symbolic regression modeling expert. This task may provide variable names,
target/feature dimensions, and numerical ranges.

Treat this as a dimension-aware mathematical structure discovery task. Build
formula priors from the supplied dimensions, variable names, and ranges.

Given the task metadata, please:
1. Extract dimensional and variable-name constraints that can guide formula structure
2. List generic mathematical formula structures suggested by dimensions, variable names, and ranges

## Output Format
Return only a JSON object:
{{"domain": "dimension-aware symbolic regression", "analysis": "<1-2 sentence dimensional/variable analysis>", "formulas": [...], {preprocessing_placeholder}"heuristics": [...]}}

- domain: Use exactly "dimension-aware symbolic regression" unless the context explicitly provides a real domain.
- analysis: Briefly explain useful dimensional relations and variable-name hints.
- formulas: List 5~15 useful **structural patterns**, requirements:
  - Use the actual variable names from the task as much as possible
  - Arrange from simple to complex
  - Prefer dimensionally plausible products, ratios, powers, square roots, sums of compatible terms, and elementary functions of dimensionless arguments
  - Include trig/log/exp forms only when applied to dimensionless variables or dimensionless combinations
  - Each entry formatted as "Structure name: formula form (dimensional or variable-name rationale)"
  - Example: "Dimensionless ratio: y = c0 * x1 / x2 (use when x1 and x2 share units)"
  - Example: "Power product: y = c0 * x1**c1 * x2**c2 (generic dimensional monomial coupling)"
  - Example: "Trigonometric dimensionless input: y = c0 * sin(c1 * theta) (angle-like or dimensionless variable)"
  - **Constant notation convention**: Constant parameters must appear in multiplicative form and must not be placed in the denominator.\
    Correct: `c0 * x`, `c0 / (x + c1)`; Incorrect: `x / c0`, `sin(x / c0)`.\
    To express "scaled variable", always write `c * var` instead of `var / c`.
"""

_DK_DIMENSION_PROMPT_FOOTER = """
- heuristics: List 5~10 generic formula structure heuristic features tied to dimensional consistency, variable names, and numerical ranges.
  Useful themes include:
  - variables with identical dimensions often appear as sums, differences, ratios, or inside dimensionless nonlinear functions
  - output dimensions can often be matched by products or ratios of input dimensions
  - dimensionless variables can safely enter sin/cos/log/exp/tanh
  - bounded positive ranges often support power, sqrt, rational, and log(1+x) structures
  - angle-like names such as theta/phi suggest trigonometric structures
  Focus on dimensionally consistent generic structures and variable-name/range cues.
"""


def _is_dimension_only_context(context_prompt: str) -> bool:
    text = (context_prompt or "").lower()
    return (
        "aifeynman:" in text
        or "no natural-language domain background" in text
        or "no natural-language background" in text
    )


def _build_dk_system_prompt(enable_preprocessing: bool, dimension_only: bool = False) -> str:
    if enable_preprocessing:
        pp_placeholder = '"preprocessing": [...], '
        if dimension_only:
            return (
                _DK_DIMENSION_PROMPT_HEADER.format(
                    preprocessing_placeholder=pp_placeholder)
                + _DK_PROMPT_PREPROCESSING
                + _DK_DIMENSION_PROMPT_FOOTER
            )
        return (
            _DK_PROMPT_HEADER.format(preprocessing_placeholder=pp_placeholder)
            + _DK_PROMPT_PREPROCESSING
            + _DK_PROMPT_FOOTER
        )
    pp_placeholder = ""
    if dimension_only:
        return (
            _DK_DIMENSION_PROMPT_HEADER.format(
                preprocessing_placeholder=pp_placeholder)
            + _DK_DIMENSION_PROMPT_FOOTER
        )
    return (
        _DK_PROMPT_HEADER.format(preprocessing_placeholder=pp_placeholder)
        + _DK_PROMPT_FOOTER
    )


DOMAIN_KNOWLEDGE_SYSTEM_PROMPT = _build_dk_system_prompt(enable_preprocessing=True)

DOMAIN_KNOWLEDGE_USER_TEMPLATE = """\
{context}

Available variables: {variables}

Please analyze the scientific domain of this task and list common formula structures related to these variables in that domain."""

DIMENSION_KNOWLEDGE_USER_TEMPLATE = """\
{context}

Available variables: {variables}

Please analyze the dimensional information, variable names, and numerical ranges,
then list useful generic formula structures for symbolic regression on this task."""

# ------------------------------------------------------------------ #
# Seed Generation Prompts (independent of the mutation prompt)
# ------------------------------------------------------------------ #

# Seeds have no parent, so they must NOT inherit any "preserve parent / add a
# term" language. This prompt is deliberately neutral about top-level shape and
# requires coverage of both monomial/product/ratio and multi-term archetypes.
SEED_SYSTEM_PROMPT = """\
You are a symbolic-regression expert. Generate diverse candidate closed-form formulas for the dataset \
described below, to seed an evolutionary search.

## Structural Prior (read carefully)
The ground-truth formula may be EITHER:
- a single compact **monomial / product / ratio / composition** -- typical of first-principles physical \
laws, e.g. `c0*x1*x2/x3`, `c0*exp(-c1*x**2)`, `c0/sqrt(1 - x1**2/c1**2)`; OR
- a **sum of several terms** -- typical of empirical / dynamical models, e.g. `c0*x + c1*x**2 + c2`, \
`c0*(c1 - x1/c2)*x1 + c3*x1**c4`.
Your seed set MUST cover **both** archetypes. Do not make every seed an additive sum, and do not make \
every seed a single product -- spread the seeds across both shapes and across simple-to-complex.

## Building Blocks
power `x**c`, trigonometric `sin`/`cos`, exponential `exp`, logarithmic `log(1+x)`, `sqrt`, and -- where \
physically meaningful -- `Abs(x)`, `Max(x, 0)`.

## Using {knowledge_basis}
**Design structures informed by the knowledge below.** Different disciplines have typical formula priors.
{domain_knowledge}

## Constant Notation Convention
Constant parameters should always appear in **multiplicative form**, never alone in a denominator: write \
`c0*x` and `c0/(x + c1)`, not `x/c0` (a constant in the denominator is numerically unstable).

## Task Background
{context}

## Available Variables
{variables}"""

SEED_SYSTEM_EXTRA = """\

Please generate {n_seeds} candidate formulas with different structures at once, \
and return only a JSON array:
[{{"expression":"...","params":["c0","c1"]}}, ...]"""

# ------------------------------------------------------------------ #
# Describer Prompts
# ------------------------------------------------------------------ #

DESCRIBER_SYSTEM_PROMPT = """\
You are a mathematical expression analysis expert. Given a symbolic formula and its fitted parameter values,
please provide a concise description of its mathematical structural features and actual behavior.

## Description Requirements
Your description should include the following information:
1. **Structural features**: Top-level structure type (additive combination/product/nesting, etc.), core mathematical components (exponential, logarithmic, rational, power, etc.), nesting/combination relationships of components
2. **Variable participation**: Position and role of each variable in the formula
3. **Parameter behavior analysis**: Based on the specific fitted parameter values, analyze the actual contribution of each term:
   - Whether terms with parameters close to 0 are effectively degenerated (e.g., exp(0.001*x)≈1)
   - Whether extremely large/small parameters suggest structural mismatch
   - Relative magnitude relationships between terms
4. **Equivalent simplification**: If parameter values cause some substructures to degenerate, indicate what simpler form the formula is approximately equivalent to

## Output Format
Return only a JSON object:
{{"description": "<a structural description text>"}}
Keep the description to 2~4 sentences, information-dense."""

DESCRIBER_USER_TEMPLATE = """\
Formula: {formula}
Fitted parameter values: {fitted_params}
Training set NMSE: {mse}"""

BATCH_DESCRIBER_SYSTEM_PROMPT = """\
You are a mathematical expression analysis expert. Given multiple symbolic formulas and their fitted parameter values,
please provide a concise description of the mathematical structural features and actual behavior for each formula separately.

## Description Requirements (required for each formula)
1. **Structural features**: Top-level structure type (additive combination/product/nesting, etc.), core mathematical components (exponential, logarithmic, rational, power, etc.), nesting/combination relationships of components
2. **Variable participation**: Position and role of each variable in the formula
3. **Parameter behavior analysis**: Based on the specific fitted parameter values, analyze the actual contribution of each term:
   - Whether terms with parameters close to 0 are effectively degenerated (e.g., exp(0.001*x)≈1)
   - Whether extremely large/small parameters suggest structural mismatch
   - Relative magnitude relationships between terms
4. **Equivalent simplification**: If parameter values cause some substructures to degenerate, indicate what simpler form the formula is approximately equivalent to

## Output Format
Return only a JSON array. Each element must correspond to one formula's description:
[
  {{"formula": "<original formula 1>", "description": "<description text>"}},
  {{"formula": "<original formula 2>", "description": "<description text>"}}
]
Keep each description to 2~4 sentences, information-dense. Array order must match input order."""

BATCH_DESCRIBER_USER_TEMPLATE = """\
Please generate structural descriptions for each of the following {count} formulas:

{entries}"""


_PARAMS_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {"type": "string"},
}

_FORMULA_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {"type": "string"},
        "params": _PARAMS_SCHEMA,
        "mutation": {"type": "string"},
    },
    "required": ["expression", "params"],
    "additionalProperties": False,
}

_DOMAIN_KNOWLEDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "domain": {"type": "string"},
        "analysis": {"type": "string"},
        "formulas": {"type": "array", "items": {"type": "string"}},
        "preprocessing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string"},
                    "type": {"type": "string", "enum": ["linear_scale", "moment"]},
                    "reason": {"type": "string"},
                },
                "required": ["variable", "type", "reason"],
                "additionalProperties": False,
            },
        },
        "heuristics": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["domain", "analysis", "formulas", "heuristics"],
    "additionalProperties": False,
}

_DESCRIBE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"description": {"type": "string"}},
    "required": ["description"],
    "additionalProperties": False,
}

_DESCRIBE_BATCH_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "formula": {"type": "string"},
            "description": {"type": "string"},
        },
        "required": ["formula", "description"],
        "additionalProperties": False,
    },
}


def _schema_for_component(component: str) -> Dict[str, Any]:
    if component == "domain_knowledge":
        return _DOMAIN_KNOWLEDGE_SCHEMA
    if component == "seed_generation":
        return {"type": "array", "items": _FORMULA_SCHEMA}
    if component == "describe":
        return _DESCRIBE_SCHEMA
    if component == "describe_batch":
        return _DESCRIBE_BATCH_SCHEMA
    return {"type": "array", "items": _FORMULA_SCHEMA}


# ------------------------------------------------------------------ #
# LLMGenerator
# ------------------------------------------------------------------ #

class LLMGenerator:
    """Domain knowledge extraction + seed formula generation + offspring description generation."""

    def __init__(
        self,
        api_client: Any,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.8,
        max_tokens: int = 512,
        usage_logger: Any = None,
        component_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        enable_preprocessing: bool = False,
    ):
        self.api = api_client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._usage_logger = usage_logger
        self._domain_knowledge: str = ""
        self._preprocessing_rules: List[Dict[str, str]] = []
        self._comp_overrides = component_overrides or {}
        self.enable_preprocessing = enable_preprocessing

    @property
    def domain_knowledge(self) -> str:
        return self._domain_knowledge

    def generate_domain_knowledge(
        self,
        context_prompt: str,
        variables: List[str],
    ) -> Tuple[str, List[Dict[str, str]]]:
        """
        Call LLM to analyze the scientific domain of the current task and generate
        domain-specific formula prior knowledge.

        Returns
        -------
        (domain knowledge text, preprocessing rules list)
        """
        dimension_only = _is_dimension_only_context(context_prompt)
        user_template = (
            DIMENSION_KNOWLEDGE_USER_TEMPLATE
            if dimension_only
            else DOMAIN_KNOWLEDGE_USER_TEMPLATE
        )
        user_prompt = user_template.format(
            context=context_prompt,
            variables=", ".join(variables),
        )
        system_prompt = _build_dk_system_prompt(
            self.enable_preprocessing,
            dimension_only=dimension_only,
        )

        for attempt in range(self.max_retries):
            try:
                raw = self._call_api(system_prompt, user_prompt,
                                     component="domain_knowledge")
                parsed = self._parse_json_response(raw)

                domain = parsed.get("domain", "")
                analysis = parsed.get("analysis", "")
                formulas = parsed.get("formulas", [])
                raw_preproc = parsed.get("preprocessing", [])
                heuristics = parsed.get("heuristics", [])

                if not formulas:
                    print(f"  [DomainKnowledge] Attempt {attempt+1} returned no formulas, retrying...")
                    continue

                preproc_rules: List[Dict[str, str]] = []
                if self.enable_preprocessing:
                    valid_types = {"linear_scale", "moment"}
                    var_set = set(variables)
                    for item in raw_preproc[:3]:
                        if isinstance(item, dict):
                            var = item.get("variable", "")
                            ptype = item.get("type", "")
                            reason = item.get("reason", "")
                            if var in var_set and ptype in valid_types:
                                preproc_rules.append({
                                    "variable": var,
                                    "type": ptype,
                                    "reason": reason,
                                })

                if dimension_only:
                    _builtin_formulas = [
                        "Dimensional monomial: y = c0 * x1**c1 * x2**c2 (match output units through products and powers)",
                        "Same-unit ratio: y = c0 * x1 / (x2 + c1) (use when two variables have compatible dimensions)",
                        "Compatible additive terms: y = c0 * x1 + c1 * x2 (use only for variables with matching dimensions after scaling)",
                        "Dimensionless nonlinear transform: y = c0 * sin(c1 * u) (use for dimensionless or angle-like variables)",
                        "Positive-range logarithm: y = c0 * log(1 + c1 * u) (use for positive dimensionless variables or scaled combinations)",
                        "Positive-range exponential: y = c0 * exp(c1 * u) (use for dimensionless inputs over moderate ranges)",
                        "Square-root product: y = c0 * sqrt(x1**2 + c1 * x2**2) (use for magnitude-like combinations of compatible variables)",
                        "Rational saturation: y = c0 * u / (u + c1) (use for positive dimensionless variables with bounded response)",
                    ]
                    formula_heading = (
                        "**Common formula structures** "
                        "(generic dimension/variable-name priors + LLM suggestions):"
                    )
                else:
                    _builtin_formulas = [
                        "Arrhenius equation: k = c0 * exp(c1 / T) (chemical reaction rate dependence on temperature)",
                        "Coulomb's / Newton's inverse-square law: F = c0 / r**2 (gravitational or electrostatic force)",
                        "Stefan-Boltzmann law: P = c0 * T**4 (blackbody radiation power vs temperature)",
                        "Beer-Lambert law: I = c0 * exp(c1 * x) (light intensity attenuation through a medium)",
                        "Michaelis-Menten kinetics: v = c0 * S / (S + c1) (enzyme-catalyzed reaction saturation)",
                        "Hill equation: y = c0 * x**c1 / (x**c1 + c2) (cooperative binding, sigmoidal dose-response)",
                        "Kepler's third law: T**2 = c0 * a**3 (orbital period vs semi-major axis)",
                        "Newton's law of cooling: T = c0 + c1 * exp(c2 * t) (exponential thermal relaxation)",
                        "Clausius-Clapeyron relation: log(P) = c0 + c1 / T (vapor pressure vs temperature)",
                        "Langmuir adsorption isotherm: theta = c0 * P / (1 + c1 * P) (surface adsorption equilibrium)",
                        "Freundlich isotherm: q = c0 * C**c1 (empirical adsorption on heterogeneous surface)",
                        "Stokes' drag: F = c0 * r * v (viscous drag force on a sphere in fluid)",
                        "Logistic growth: N = c0 / (1 + c1 * exp(c2 * t)) (population dynamics with carrying capacity)",
                        "Planck-Wien approximation: I = c0 * f**3 * exp(c1 * f / T) (thermal radiation spectrum)",
                    ]
                    formula_heading = (
                        "**Common formula structures** "
                        "(built-in generic + LLM domain-specific):"
                    )
                all_formulas = _builtin_formulas + (formulas or [])

                lines = [
                    "\n## Dimension/Variable Knowledge" if dimension_only else "\n## Domain Knowledge",
                    f"**Domain**: {domain}" if domain else "",
                    f"**Analysis**: {analysis}" if analysis else "",
                    "",
                    formula_heading,
                ]
                lines = [l for l in lines if l or l == ""]
                for f in all_formulas:
                    lines.append(f"- {f}")

                if preproc_rules:
                    lines.append("")
                    lines.append("**Variable preprocessing rules**:")
                    type_desc = {
                        "linear_scale": "Linear scaling x→c1*x+c2",
                        "moment": "Moment transform x→x^c1",
                    }
                    for r in preproc_rules:
                        desc = type_desc.get(r["type"], r["type"])
                        lines.append(
                            f"- {r['variable']}: {desc} — {r['reason']}")

                if dimension_only:
                    _builtin_heuristics = [
                        "Dimensional matching → Products, ratios, and powers can combine input units to match the target units",
                        "Dimensionless arguments → sin/cos/log/exp/tanh should receive dimensionless variables or dimensionless combinations",
                        "Same-unit variables → Sums, differences, ratios, and Euclidean magnitudes are often plausible",
                        "Positive finite ranges → sqrt, power, rational, and log(1+x) structures are numerically plausible",
                        "Angle-like variable names → theta/phi or dimensionless angular variables can justify trigonometric terms",
                    ]
                    heuristic_heading = (
                        "**Formula structure heuristic features** "
                        "(dimension/variable-name based):"
                    )
                else:
                    _builtin_heuristics = [
                        "Oscillation damping envelope → Exponential decay envelope f(x)*exp(-c*Abs(x)) for symmetric damped oscillation, Gaussian envelope f(x)*exp(-c*x**2) for localized wave packets",
                        "Saturation/clipping effects → Rational fraction x/(1+x), tanh; for variables with alternating positive/negative in oscillatory systems, selectively apply Abs",
                        "Growth/decay → Exponential growth exp(c*t), Logistic 1/(1+exp(-c*x)), power law x**c",
                        "Threshold/activation behavior → Max(x-c, 0), piecewise functions",
                        "Coupling/cross effects → Product terms between variables x*y, power-law coupling x**c1 * y**c2",
                    ]
                    heuristic_heading = "**Formula structure heuristic features**:"
                all_heuristics = _builtin_heuristics + (heuristics or [])
                lines.append("")
                lines.append(heuristic_heading)
                for h in all_heuristics:
                    lines.append(f"- {h}")

                lines.append("")

                self._domain_knowledge = "\n".join(lines)
                self._preprocessing_rules = preproc_rules
                print(f"  [DomainKnowledge] Successfully generated domain knowledge: {domain}")
                if preproc_rules:
                    print(f"  [DomainKnowledge] Preprocessing rules: "
                          f"{[r['variable']+'→'+r['type'] for r in preproc_rules]}")
                return self._domain_knowledge, self._preprocessing_rules

            except Exception as e:
                print(f"  [DomainKnowledge] Attempt {attempt+1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)

        self._domain_knowledge = ""
        self._preprocessing_rules = []
        return "", []

    def initialize_seeds(
        self,
        context_prompt: str,
        variables: List[str],
        n_seeds: int = 20,
    ) -> List[Dict[str, Any]]:
        """Let LLM generate initial seed formula list based on task metadata."""
        dimension_only = _is_dimension_only_context(context_prompt)
        knowledge_basis = (
            "Dimensional Information, Variable Names & Generic Structures"
            if dimension_only
            else "Domain Knowledge"
        )
        system_prompt = SEED_SYSTEM_PROMPT.format(
            context=context_prompt,
            domain_knowledge=self._domain_knowledge,
            variables=", ".join(variables),
            knowledge_basis=knowledge_basis,
        ) + SEED_SYSTEM_EXTRA.format(n_seeds=n_seeds)
        if dimension_only:
            category_b = (
                "**B. Dimension/variable-informed formulas** (approximately half): "
                "Based on dimensional information, variable-name hints, numerical ranges, "
                "and formula structures listed above, generate plausible generic symbolic "
                "structures."
            )
        else:
            category_b = (
                "**B. Domain knowledge formulas** (approximately half): Based on the task "
                "background and formulas listed in the domain knowledge section, generate "
                "classical formula forms most likely to match the data in this domain. "
                "These formulas should reflect domain-specific mathematical structures, "
                "not generic functions."
            )
        user_prompt = (
            f"Based on the above task metadata, generate {n_seeds} candidate formulas with diverse structures.\n\n"
            f"## Requirements\n"
            f"Formulas must cover both of the following categories, arranged from simple to complex:\n\n"
            f"**A. Basic function structures** (approximately half): Ensure at least one variant of each basic type. "
            f"Include BOTH compact single-term shapes (monomial/product/ratio/composition) AND multi-term sums:\n"
            f"- Monomial / product: c0*x1*x2, c0*x1*x2/x3 (couple variables multiplicatively, not only additively)\n"
            f"- Power-law / composition: c0*x**c1, c0*exp(-c1*x**2), c0/sqrt(1 - x**2/c1**2)\n"
            f"- Linear: c0*x + c1\n"
            f"- Quadratic/polynomial: c0*x**2 + c1*x + c2\n"
            f"- Non-integer power law: c0*x**c1\n"
            f"- Exponential: c0*exp(c1*x)\n"
            f"- Logarithmic: c0*log(1 + c1*x)\n"
            f"- Rational: c0*x/(x + c1)\n"
            f"- Trigonometric: c0*sin(c1*x)\n"
            f"- Hyperbolic: c0*tanh(c1*x + c2)\n"
            f"- Square root: c0*sqrt(x**2 + c1)\n"
            f"- Simple combinations of the above basic types (e.g., polynomial+exponential, rational+linear, etc.)\n\n"
            f"{category_b}\n\n"
            f"Note: All constants in formulas must be represented as c0, c1, c2..., and variables must use names from the available variables list."
        )

        valid_seeds: List[Dict[str, Any]] = []
        seen_exprs = set()

        def _add_seed(seed: Dict[str, Any]) -> bool:
            expr_key = " ".join(seed.get("expression", "").strip().split())
            if not expr_key or expr_key in seen_exprs:
                return False
            seen_exprs.add(expr_key)
            valid_seeds.append(seed)
            return True

        for attempt in range(self.max_retries):
            try:
                raw = self._call_api(system_prompt, user_prompt,
                                     component="seed_generation")
                parsed = self._parse_json_array_response(raw)
                valid = [p for p in parsed if self._validate_response(p, variables)]
                for seed in valid:
                    _add_seed(seed)
                if len(valid_seeds) >= n_seeds:
                    return valid_seeds[:n_seeds]
                if attempt < self.max_retries - 1:
                    print(
                        f"  [Generator] Seed generation attempt {attempt+1} produced "
                        f"{len(valid_seeds)}/{n_seeds} unique valid seeds, retrying..."
                    )
                else:
                    print(
                        f"  [Generator] Seed generation produced "
                        f"{len(valid_seeds)}/{n_seeds} unique valid seeds; "
                        "supplementing with fallback seeds."
                    )
            except Exception as e:
                print(f"  [Generator] Seed generation attempt {attempt+1} failed: {e}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        fallback_added = 0
        for seed in self._fallback_seeds(variables, n_seeds):
            if len(valid_seeds) >= n_seeds:
                break
            if _add_seed(seed):
                fallback_added += 1
        if fallback_added or len(valid_seeds) < n_seeds:
            print(
                f"  [Generator] Used {fallback_added} fallback seed suggestions "
                f"after retries ({len(valid_seeds)}/{n_seeds} total)."
            )
        return valid_seeds[:n_seeds]

    def describe(
        self,
        formula: str,
        fitted_params_str: str,
        mse: float,
    ) -> str:
        """Call LLM to generate posterior structural description after fitting. Returns empty string on failure."""
        nmse_str = f"{mse:.2e}" if mse < 1e9 else "Not evaluated"
        user_prompt = DESCRIBER_USER_TEMPLATE.format(
            formula=formula,
            fitted_params=fitted_params_str or "No constant parameters",
            mse=nmse_str,
        )
        try:
            raw = self._call_api(DESCRIBER_SYSTEM_PROMPT, user_prompt,
                                 component="describe")
            parsed = self._parse_json_response(raw)
            desc = parsed.get("description", "").strip()
            if desc:
                return desc
        except Exception as e:
            print(f"  [Describer] Description generation failed: {e}")
        return ""

    def describe_batch(
        self,
        items: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate structural descriptions in batch."""
        if not items:
            return []
        if len(items) == 1:
            desc = self.describe(
                formula=items[0]["formula"],
                fitted_params_str=items[0]["fitted_params_str"],
                mse=items[0]["mse"],
            )
            return [desc]

        entries_parts = []
        for i, item in enumerate(items):
            nmse_str = f"{item['mse']:.2e}" if item["mse"] < 1e9 else "Not evaluated"
            entries_parts.append(
                f"### Formula {i+1}\n"
                f"Formula: {item['formula']}\n"
                f"Fitted parameter values: {item['fitted_params_str'] or 'No constant parameters'}\n"
                f"Training set NMSE: {nmse_str}"
            )

        user_prompt = BATCH_DESCRIBER_USER_TEMPLATE.format(
            count=len(items),
            entries="\n\n".join(entries_parts),
        )

        try:
            raw = self._call_api(BATCH_DESCRIBER_SYSTEM_PROMPT, user_prompt,
                                 component="describe_batch")
            parsed_list = self._parse_json_array_response(raw)

            formula_to_desc = {}
            for entry in parsed_list:
                f = entry.get("formula", "").strip()
                d = entry.get("description", "").strip()
                if f and d:
                    formula_to_desc[f] = d

            results = []
            for i, item in enumerate(items):
                desc = formula_to_desc.get(item["formula"], "")
                if not desc and i < len(parsed_list):
                    desc = parsed_list[i].get("description", "").strip()
                results.append(desc)
            return results

        except Exception as e:
            print(f"  [Describer] Batch description generation failed: {e}, falling back to individual...")
            return [
                self.describe(
                    formula=item["formula"],
                    fitted_params_str=item["fitted_params_str"],
                    mse=item["mse"],
                )
                for item in items
            ]

    # ------------------------------------------------------------------ #
    # Internal utilities
    # ------------------------------------------------------------------ #

    def _call_api(self, system_prompt: str, user_prompt: str,
                  component: str = "generator") -> str:
        import time as _t
        ov = self._comp_overrides.get(component, {})
        if not ov and component == "describe":
            ov = self._comp_overrides.get("describe_batch", {})
        api_client  = ov.get("api_client", self.api)
        model       = ov.get("model", self.model)
        temperature = ov.get("temperature", self.temperature)
        max_tokens  = ov.get("max_tokens", self.max_tokens)
        messages = with_json_only_instruction([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        completion_max_tokens = resolve_completion_max_tokens(
            model, messages, max_tokens)
        
        # ✅ Extract additional kwargs from component overrides (e.g., anthropic_version)
        create_kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": completion_max_tokens,
        }
        for key in ov:
            if key not in ("api_client", "model", "temperature", "max_tokens"):
                create_kwargs[key] = ov[key]
        add_guided_json_schema(
            create_kwargs,
            _schema_for_component(component),
            api_client,
        )
        
        t0 = _t.monotonic()
        response = chat_completion(api_client, create_kwargs, component=component)
        elapsed = _t.monotonic() - t0
        if self._usage_logger and response.usage:
            self._usage_logger.log(
                component=component,
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                duration_s=elapsed,
            )
        return response.choices[0].message.content.strip()

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        text = _extract_code_block(raw)
        result = json_repair.loads(text)
        if isinstance(result, dict):
            return result
        if isinstance(result, list) and result and isinstance(result[0], dict):
            return result[0]
        raise ValueError(f"Failed to parse JSON object from response: {raw[:200]}")

    def _parse_json_array_response(self, raw: str) -> List[Dict[str, Any]]:
        text = _extract_code_block(raw)
        result = json_repair.loads(text)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        if isinstance(result, dict):
            return [result]
        return []

    @staticmethod
    def _validate_response(parsed: Dict[str, Any], variables: List[str]) -> bool:
        if not isinstance(parsed, dict):
            return False
        expr = parsed.get("expression", "")
        if not isinstance(expr, str) or not expr.strip() or len(expr) > 500:
            return False
        params = parsed.get("params", [])
        if not isinstance(params, list):
            return False
        if not all(isinstance(p, str) and re.fullmatch(r"c\d+", p) for p in params):
            return False
        if len(set(params)) != len(params):
            return False
        var_set = set(variables)
        locals_map = {name: sp.Symbol(name) for name in var_set | set(params)}
        try:
            sp_expr = sp.sympify(expr, locals=locals_map)
        except Exception:
            return False
        if not isinstance(sp_expr, sp.Expr):
            return False
        free_symbols = {str(sym) for sym in sp_expr.free_symbols}
        if not free_symbols <= (var_set | set(params)):
            return False
        used_params = {name for name in free_symbols if re.fullmatch(r"c\d+", name)}
        if used_params != set(params):
            return False
        return True

    @staticmethod
    def _format_top_exprs(
        top_exprs: Optional[List[Dict[str, Any]]], topk: int
    ) -> str:
        if not top_exprs:
            return "(No history yet)"
        lines = []
        for i, item in enumerate(top_exprs[:topk], 1):
            train = item.get("train_nmse") or float("inf")
            train_str = f"{train:.2e}" if isinstance(train, (int, float)) and train < 1e9 else "∞"
            params_str = item.get("fitted_params", "")
            params_part = f"  Params: {params_str}" if params_str else ""
            lines.append(
                f"  #{i}: {item.get('expression', '?')}"
                f"  (train_nmse={train_str}){params_part}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fallback_seeds(variables: List[str], n: int) -> List[Dict[str, Any]]:
        v = variables[0] if variables else "x"
        seeds = [
            {"expression": f"c0 * {v} + c1", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v}**2 + c1 * {v} + c2", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * {v}**c1", "params": ["c0", "c1"]},
            {"expression": f"c0 * exp(c1 * {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * log(1 + c1 * {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v} / ({v} + c1)", "params": ["c0", "c1"]},
            {"expression": f"c0 * sin(c1 * {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * tanh(c1 * {v} + c2)", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * sqrt({v}**2 + c1)", "params": ["c0", "c1"]},
            {"expression": f"c0 / (c1 + {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v}**2 + c1 * exp(c2 * {v})", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * {v} * exp(c1 * {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v} / ({v}**2 + c1)", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v}**3 + c1 * {v}**2 + c2 * {v} + c3", "params": ["c0", "c1", "c2", "c3"]},
            {"expression": f"c0 * exp(c1 / ({v} + c2))", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * {v} * log(1 + c1 * {v})", "params": ["c0", "c1"]},
            {"expression": f"c0 * (1 - exp(c1 * {v}))", "params": ["c0", "c1"]},
            {"expression": f"c0 * {v}**c1 + c2", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * sin(c1 * {v}) + c2 * {v}", "params": ["c0", "c1", "c2"]},
            {"expression": f"c0 * {v} / (c1 + exp(c2 * {v}))", "params": ["c0", "c1", "c2"]},
        ]
        return seeds[:n]


# ------------------------------------------------------------------ #
# Mock Generator
# ------------------------------------------------------------------ #

class MockGenerator:
    """Mock Generator: does not call LLM."""

    def __init__(self, variables: List[str], seed: int = 42):
        self.variables = variables
        self._domain_knowledge = ""

    @property
    def domain_knowledge(self) -> str:
        return self._domain_knowledge

    def generate_domain_knowledge(self, **kwargs) -> Tuple[str, List[Dict[str, str]]]:
        return "", []

    def initialize_seeds(self, variables: List[str] = None, **kwargs) -> List[Dict[str, Any]]:
        return LLMGenerator._fallback_seeds(
            variables or self.variables or ["x"],
            kwargs.get("n_seeds", 20),
        )

    def describe(self, **kwargs) -> str:
        return ""

    def describe_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        return [""] * len(items)


# ------------------------------------------------------------------ #
# Factory function
# ------------------------------------------------------------------ #

def create_generator(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 512,
    max_retries: int = 3,
    usage_logger: Any = None,
    component_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    enable_preprocessing: bool = False,
    llm_mode: str = "openai",
) -> LLMGenerator:
    """Create a Generator LLM agent."""
    client = build_openai_client(model, base_url, mode=llm_mode, api_key=api_key)
    return LLMGenerator(
        api_client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        usage_logger=usage_logger,
        component_overrides=component_overrides,
        enable_preprocessing=enable_preprocessing,
    )
