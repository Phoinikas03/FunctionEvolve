#!/usr/bin/env python3
"""
Verify whether candidate models (formulas or Python code) are mathematically equivalent to the GT expression.
The overall conclusion is positive if any candidate matches, and the output also
includes the full list of matching candidates.

Supports three input methods for the candidates:
  1. Legacy positional log path (Default): python verify.py <log_path> [--llm-config config.yaml]
     (Extracts <<<FORMULA>>> by default, also supports <<<CODE>>> and legacy log formats)
  2. Directly reading a .py file: python verify.py --py <file.py> [--eq-name PO_1]
  3. Directly passing a string: python verify.py --str "c0*X + c1" --eq-name PO_1

Usage Examples:
    python verify.py experiment_PO_1.log
    python verify.py --py my_model.py --eq-name BPG_2
    python verify.py --str "def f(x): return x**2" --eq-name CRK_5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Optional

GT_CSV = Path(__file__).parent / "datasets" / "llm-srbench" / "gt_expressions.csv"
FEYNMAN_CSV = Path(__file__).parent / "datasets" / "aifeynman" / "FeynmanEquations.csv"
BONUS_CSV = Path(__file__).parent / "datasets" / "aifeynman" / "BonusEquations.csv"

PO_X_WITH_NEGATIVE = {
    "PO0", "PO1", "PO2", "PO4", "PO5", "PO6", "PO7", "PO8", "PO9",
    "PO10", "PO11", "PO12", "PO13", "PO14", "PO15", "PO16", "PO17",
    "PO18", "PO19", "PO20", "PO23", "PO26", "PO28", "PO29", "PO30",
    "PO31", "PO32", "PO33", "PO34", "PO35", "PO37", "PO38", "PO40",
    "PO41", "PO43",
}
PO_X_POSITIVE = {"PO21", "PO22", "PO27", "PO36", "PO39", "PO42"}
PO_X_MISSING = {"PO3", "PO24", "PO25"}

PO_V_WITH_NEGATIVE = {
    "PO0", "PO2", "PO3", "PO4", "PO5", "PO6", "PO7", "PO8", "PO9",
    "PO11", "PO12", "PO13", "PO14", "PO15", "PO16", "PO17", "PO18",
    "PO19", "PO20", "PO21", "PO22", "PO23", "PO24", "PO25", "PO26",
    "PO27", "PO28", "PO29", "PO30", "PO31", "PO32", "PO33", "PO34",
    "PO35", "PO36", "PO37", "PO38", "PO39", "PO40", "PO41", "PO42",
    "PO43",
}
PO_V_WITH_BOTH_SIGNS = PO_V_WITH_NEGATIVE - {"PO22", "PO23"}
PO_V_NONPOSITIVE = {"PO22", "PO23"}
PO_V_MISSING = {"PO1", "PO10"}


def extract_equation_name(file_path: str) -> str:
    """Extract the equation name from file content or filename."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"equation=([\w.]+)", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    
    basename = Path(file_path).stem
    return basename.split("_")[0]


def extract_dataset_name(file_path: str) -> str:
    """Extract the dataset name (e.g. 'aifeynman', 'llm-srbench') from file content or path."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                m = re.search(r"dataset=([\w-]+)", line)
                if m:
                    return m.group(1)
    except Exception:
        pass
    # Fallback: infer from path
    path_str = str(file_path)
    if "aifeynman" in path_str:
        return "aifeynman"
    return "llm-srbench"


def load_gt(equation_name: str, dataset: str = "llm-srbench") -> Optional[dict]:
    """Load GT info from the appropriate CSV based on dataset."""
    if dataset == "aifeynman":
        return _load_gt_aifeynman(equation_name)

    if not GT_CSV.is_file():
        return None
        
    with open(GT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["equation_name"] == equation_name:
                return row
    return None


def _load_gt_aifeynman(equation_name: str) -> Optional[dict]:
    """Load GT info from FeynmanEquations.csv or BonusEquations.csv."""
    for csv_path in (FEYNMAN_CSV, BONUS_CSV):
        if not csv_path.is_file():
            continue
        with open(csv_path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("Filename") == equation_name:
                    # Convert Feynman CSV format to the dict format verify expects
                    n_vars = int(row.get("# variables", 0))
                    feature_names = ";".join(
                        row[f"v{i}_name"] for i in range(1, n_vars + 1)
                        if row.get(f"v{i}_name")
                    )
                    return {
                        "equation_name": equation_name,
                        "symbolic_expression": row.get("Formula", ""),
                        "feature_names": feature_names,
                        "param_names": "",
                        "gt_params": "",
                        "numerical_expression": row.get("Formula", ""),
                    }
    return None


def canonical_po_name(eq_name: str) -> Optional[str]:
    """Return canonical PO case name, e.g. PO36, when eq_name refers to one."""
    match = re.search(r"\bPO_?(\d+)\b", eq_name)
    if not match:
        return None
    return f"PO{int(match.group(1))}"


def build_po_domain_rule(eq_name: str) -> str:
    """Describe the observed benchmark-domain sign constraints for a PO case."""
    po_name = canonical_po_name(eq_name)
    if po_name is None:
        return """- **Domain Assumption (Physics Oscillators):** Use the observed benchmark domain when judging absolute-value equivalences. Do NOT assume a variable is non-negative unless a PO-specific rule below says so."""

    x_rule = "x is not present in this case."
    if po_name in PO_X_POSITIVE:
        x_rule = "`x` is strictly positive on this benchmark case; `Abs(x)` is equivalent to `x`."
    elif po_name in PO_X_WITH_NEGATIVE:
        x_rule = "`x` takes both negative and positive values; `Abs(x)` is NOT equivalent to `x` or `-x`."
    elif po_name not in PO_X_MISSING:
        x_rule = "`x` has no explicit sign rule; do NOT simplify `Abs(x)`."

    v_rule = "v is not present in this case."
    if po_name in PO_V_WITH_BOTH_SIGNS:
        v_rule = "`v` takes both negative and positive values; `Abs(v)` is NOT equivalent to `v` or `-v`."
    elif po_name in PO_V_NONPOSITIVE:
        v_rule = "`v` is non-positive on this benchmark case; `Abs(v)` is equivalent to `-v`."
    elif po_name not in PO_V_MISSING:
        v_rule = "`v` has no explicit sign rule; do NOT simplify `Abs(v)`."

    return f"""- **PO Benchmark Domain (case-specific):** Judge equivalence on the observed benchmark domain for {po_name}, not on a generic unconstrained oscillator domain.
  - `t` is non-negative for all PO cases; `Abs(t)` is equivalent to `t`.
  - {x_rule}
  - {v_rule}
  - For variables with both negative and positive values, keep absolute values distinct. For variables constrained to one sign in this specific PO case, use that sign constraint when judging equivalence."""


def extract_candidates_from_file(file_path: str) -> list[str]:
    """Extract all candidate formulas or code snippets from the log/text file."""
    text = Path(file_path).read_text(encoding="utf-8")

    # 1. <<<FORMULA>>> tag (Highest Priority for text logs)
    tagged_formulas = re.findall(r"<<<FORMULA>>>(.+?)<<<END_FORMULA>>>", text, re.DOTALL)
    if tagged_formulas:
        return [f.strip() for f in tagged_formulas if f.strip()]

    # 2. <<<CODE>>> tag
    tagged_codes = re.findall(r"<<<CODE>>>(.+?)<<<END_CODE>>>", text, re.DOTALL)
    if tagged_codes:
        return [c.strip() for c in tagged_codes if c.strip()]

    # 3. Markdown python block tag
    md_codes = re.findall(r"```python\s*(.+?)```", text, re.DOTALL)
    if md_codes:
        return [c.strip() for c in md_codes if c.strip()]

    # 4. Legacy Block Extraction logic
    lines = text.splitlines()
    pattern_numbered = re.compile(r"^\s*\d+\.\s+(?:\[.+?\]\s+)?(\S.+)$")

    def _extract_block(start: int) -> list[str]:
        result = []
        for line in lines[start:]:
            if line.startswith("=" * 10):
                break
            m = pattern_numbered.match(line)
            if m:
                formula = m.group(1).strip()
                if not formula.startswith("param") and not formula.startswith("train"):
                    result.append(formula)
        return result

    for i, line in enumerate(lines):
        if re.match(r"Final output\s+\d+\s+formulas", line):
            formulas = _extract_block(i + 1)
            if formulas:
                return formulas

    for i, line in enumerate(lines):
        if re.match(r"Mature nodes\s*\(\d+\)", line):
            formulas = _extract_block(i + 1)
            if formulas:
                return formulas

    # 5. Single line/block Best Expression or Code fallback
    for i, line in enumerate(lines):
        m = re.match(r"Best\s+(?:expression|formula|code|program|python)\s*[:：]\s*(.*)", line, re.IGNORECASE)
        if m:
            single_line = m.group(1).strip()
            if single_line:
                return [single_line]
            else:
                # Might be a multi-line code block below the indicator
                snippet = "\n".join(lines[i+1:]).strip()
                if snippet:
                    return [snippet]

    return []


def build_prompt(
    gt_symbolic: str,
    gt_features: str,
    candidates: list[str],
    eq_name: str,
    param_names: str = "",
    gt_params: str = "",
    numerical_expression: str = "",
) -> str:
    features = gt_features.replace(";", ", ")
    
    candidate_list = "\n\n".join(
        f"### Candidate {i+1}:\n{cand}" 
        for i, cand in enumerate(candidates)
    )

    gt_numerical_section = ""
    if numerical_expression:
        gt_numerical_section += f"\n- **GT numerical expression** (constants substituted): `{numerical_expression}`"
    if param_names and gt_params:
        params = param_names.split(";")
        values = gt_params.split(";")
        if len(params) == len(values):
            param_list = ", ".join(f"{p}={v}" for p, v in zip(params, values))
            gt_numerical_section += f"\n- **GT parameter values**: {param_list}"

    if canonical_po_name(eq_name):
        domain_rule = build_po_domain_rule(eq_name)
    else:
        domain_rule = """- **Conditional Equivalence (Relaxed Domain - Non-Negative Variables):** For this class of problems (e.g., concentration, pressure, temperature), variables are typically strict physical quantities that are non-negative (X >= 0). Under this explicit condition, `Abs(X)` or `|X|` CAN be perfectly matched to `X`. Use this domain assumption to resolve absolute value differences, but ONLY for absolute values."""

    return f"""You are an expert mathematician, Python developer, and symbolic regression judge. Your task is to verify which candidates mathematically equate to the Ground Truth (GT).

## Core Principles
- **Representation:** Candidates may be symbolic mathematical formulas or Python code snippets. If a candidate is Python code, first deduce the mathematical formula implemented by its return value (handling libraries like `np.exp`, `math.sin`, and intermediate variables). If it is a symbolic formula, analyze it directly.
- **GT constants are FIXED.** The GT symbolic expression uses placeholders (c0, c1...), but their exact numerical values are provided below. Treat the GT as a concrete, immutable function.
- **Candidate constants are FREE.** You can assign ANY real number (including 0) to candidate constants (e.g., c0, c1...) or freely modify explicit numbers in the candidate.

## Strict Algebraic & Logical Rules (CRITICAL)
1. **Rational Function Equivalence (Common Denominators):** Do NOT reject formulas based on surface-level structure. A polynomial plus a simple fraction can perfectly equal a complex fraction. **You MUST attempt to find a common denominator (通分).**
   * *Example:* `c0*X + c1*X/(X+c2)` can perfectly match `c3*X**2/(X+c4)` by setting `c1 = -c0*c2`.
2. **NO Asymptotic Approximations:** Mathematical equivalence means EXACT equivalence across the continuous domain. You are STRICTLY FORBIDDEN from using limits or dropping constants.
   * *Example:* `log(c0*exp(X) + 1)` is a Softplus function. It is NEVER strictly equivalent to a linear function like `a*X + b`. Do not drop the `+ 1`.
3. **Linear Independence of Basis Functions:** Polynomials, exponentials, and trigonometric functions are linearly independent. 
   * *Example:* A combination of `X` and `sin(X)` can NEVER exactly equal `X**3` for all `X`. If a required basis function is missing, it is a `[NO MATCH]`.
4. **Trigonometric Equivalence:** Phase shifts matter. `sin(X + c)` can perfectly match `cos(X)` by setting `c = pi/2`.
5. **Zeroing Constants:** If a candidate has extra unwanted terms (e.g., an explicit time dependence `t` when GT is static), you can set their coefficients to 0 to eliminate them.

## Ground Truth Information
- **Equation Category/Name**: {eq_name}
- **Feature variables**: {features}
- **GT symbolic expression**: {gt_symbolic}{gt_numerical_section}

## Domain Constraints
{domain_rule}

## Candidate List
{candidate_list}

## Output Format
First, analyze each candidate step by step. Use algebraic manipulation (like finding common denominators) where necessary to test equivalence. Explicitly verify if basis functions match. 
For every candidate, include a separate line exactly in this format: `Candidate N verdict: [MATCH]` or `Candidate N verdict: [NO MATCH]`.
After all candidate analyses, include a separate line exactly in this format: `Matched candidate IDs: [1, 3]` (or `Matched candidate IDs: []` if none match).
Then, on the **very last line** of your response, output exactly `[MATCH FOUND]` if at least one candidate matches, otherwise `[NO MATCH]`. Do not add punctuation after the bracketed verdict."""


def call_llm(prompt: str, llm_config: dict) -> str:
    """Call LLM and get response."""
    gen_cfg = llm_config.get("generator", {})
    model = gen_cfg.get("model", "")
    base_url = gen_cfg.get("base_url")
    api_key = gen_cfg.get("api_key")
    llm_mode = gen_cfg.get("mode", "openai")
    
    # CRITICAL: equivalence judging must be deterministic; ignore yaml temperature
    temperature = 0.0
    max_tokens = gen_cfg.get("max_tokens", 8192)
    anthropic_version = gen_cfg.get("anthropic_version")
    reasoning_effort = gen_cfg.get("reasoning_effort")

    from src.llm_client import build_openai_client, resolve_completion_max_tokens
    client = build_openai_client(model, base_url, mode=llm_mode, api_key=api_key)

    messages = [{"role": "user", "content": prompt}]
    request_kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": resolve_completion_max_tokens(model, messages, max_tokens),
    }
    
    if anthropic_version:
        request_kwargs["anthropic_version"] = anthropic_version
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    response = client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content


def parse_llm_verdict(answer: str) -> str:
    """Parse the final LLM verdict from common bracketed tag variants."""
    # Models sometimes Markdown-escape the brackets as \[MATCH FOUND\].
    tag_pattern = re.compile(
        r"\\?\[\s*(MATCH\s+FOUND|NO\s+MATCH)\s*\\?\]",
        re.IGNORECASE,
    )
    tag_matches = list(tag_pattern.finditer(answer))
    if tag_matches:
        verdict = re.sub(r"\s+", " ", tag_matches[-1].group(1).strip()).lower()
        if verdict == "no match":
            return "No match"
        if verdict == "match found":
            return "Match found"

    # Fallback for older/non-compliant responses that spell out a conclusion.
    conclusion_pattern = re.compile(
        r"^\s*conclusion\s*[:：]\s*(match\s+found|no\s+match)\b",
        re.IGNORECASE | re.MULTILINE,
    )
    conclusion_matches = list(conclusion_pattern.finditer(answer))
    if conclusion_matches:
        verdict = re.sub(r"\s+", " ", conclusion_matches[-1].group(1).strip()).lower()
        if verdict == "no match":
            return "No match"
        if verdict == "match found":
            return "Match found"

    return "Unable to parse (LLM did not return explicit tag)"


def parse_matched_candidate_ids(answer: str, n_candidates: int) -> list[int]:
    """Parse 1-based candidate IDs marked as matching by the LLM."""
    candidate_verdict_pattern = re.compile(
        r"^\s*Candidate\s+(\d+)\s+verdict\s*[:：]\s*\\?\[\s*(MATCH|NO\s+MATCH)\s*\\?\]",
        re.IGNORECASE | re.MULTILINE,
    )
    ids: list[int] = []
    for match in candidate_verdict_pattern.finditer(answer):
        verdict = re.sub(r"\s+", " ", match.group(2).strip()).lower()
        if verdict == "match":
            ids.append(int(match.group(1)))

    if not ids:
        matched_ids_pattern = re.compile(
            r"^\s*Matched\s+candidate\s+IDs?\s*[:：]\s*(\[[^\]\n]*\])",
            re.IGNORECASE | re.MULTILINE,
        )
        matched_ids = list(matched_ids_pattern.finditer(answer))
        if matched_ids:
            ids.extend(int(value) for value in re.findall(r"\d+", matched_ids[-1].group(1)))

    valid_ids: list[int] = []
    seen: set[int] = set()
    for candidate_id in ids:
        if 1 <= candidate_id <= n_candidates and candidate_id not in seen:
            valid_ids.append(candidate_id)
            seen.add(candidate_id)
    return valid_ids


def verify_candidates(eq_name: str, candidates: list[str], llm_config: dict, verbose: bool = True, dataset: str = "llm-srbench") -> str:
    """Core verification function decoupled from file reading."""
    def _print(msg: str):
        if verbose:
            print(msg)

    _print(f"[verify] Equation name: {eq_name}")

    gt = load_gt(eq_name, dataset=dataset)
    if gt is None:
        return f"[verify error] Equation '{eq_name}' not found in gt_expressions.csv"

    gt_symbolic = gt["symbolic_expression"]
    gt_features = gt["feature_names"]
    _print(f"[verify] GT symbolic expression: {gt_symbolic}")
    _print(f"[verify] Feature variables: {gt_features}")
    _print(f"[verify] Processing {len(candidates)} candidate(s)")

    gt_param_names = gt.get("param_names", "")
    gt_params = gt.get("gt_params", "")
    gt_numerical = gt.get("numerical_expression", "")

    prompt = build_prompt(
        gt_symbolic, gt_features, candidates, eq_name,
        param_names=gt_param_names,
        gt_params=gt_params,
        numerical_expression=gt_numerical,
    )
    
    _print(f"[verify] Calling LLM for equivalence check...")

    try:
        answer = call_llm(prompt, llm_config)
    except Exception as e:
        return f"[verify error] LLM call failed: {e}"

    conclusion = parse_llm_verdict(answer)
    matched_candidate_ids = parse_matched_candidate_ids(answer, len(candidates))
    if conclusion == "No match":
        matched_candidate_ids = []
    elif conclusion == "Match found" and not matched_candidate_ids and len(candidates) == 1:
        matched_candidate_ids = [1]
    matched_candidates = [candidates[i - 1] for i in matched_candidate_ids]

    lines = [
        "",
        "=" * 60,
        f"GT Equivalence Verification ({eq_name})",
        "=" * 60,
        f"GT: {gt_symbolic}",
        f"Candidates evaluated: {len(candidates)}",
        f"Matched candidate IDs: {matched_candidate_ids}",
        f"Matched candidates: {json.dumps(matched_candidates, ensure_ascii=False)}",
        "",
        answer,
        "",
        f"Conclusion: {conclusion}",
        "=" * 60,
    ]
    result_text = "\n".join(lines)

    if verbose:
        print(result_text)

    return result_text


def main():
    parser = argparse.ArgumentParser(
        description="Verify whether candidate model output (formula or code) is equivalent to GT"
    )
    
    # 1. Legacy positional argument (Fully backward compatible)
    parser.add_argument("legacy_log_path", type=str, nargs="?", default=None, 
                        help="Path to the log file (Legacy positional argument)")
    
    # 2. Alternative input sources
    parser.add_argument("--log", type=str, help="Path to the log file (Explicit flag)")
    parser.add_argument("--py", type=str, help="Path to a .py file containing the code")
    parser.add_argument("--str", type=str, help="A string containing the formula or Python code")
    
    parser.add_argument("--eq-name", type=str, help="Equation name (required for --str, optional override for others)")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g. aifeynman, llm-srbench). Auto-detected from log if omitted.")
    parser.add_argument("--llm-config", type=str, default="llm_config.yaml",
                        help="Path to LLM config YAML file (default: llm_config.yaml)")
    
    args = parser.parse_args()

    target_log = args.log or args.legacy_log_path
    provided_inputs = [bool(target_log), bool(args.py), bool(args.str)]
    
    if sum(provided_inputs) == 0:
        parser.print_help()
        print("\nError: You must provide an input source: a log file (positional or --log), --py, or --str.")
        sys.exit(1)
        
    if sum(provided_inputs) > 1:
        print("Error: You can only specify ONE input method: log file, --py, or --str.")
        sys.exit(1)

    import yaml
    cfg_path = Path(args.llm_config)
    if not cfg_path.is_file():
        print(f"Error: LLM config file does not exist: {cfg_path}")
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f) or {}

    candidates = []
    eq_name = args.eq_name
    dataset = args.dataset

    # Mode 1: Log/Text file (Positional or --log)
    if target_log:
        if not Path(target_log).is_file():
            print(f"Error: Log file does not exist: {target_log}")
            sys.exit(1)
        if not eq_name:
            eq_name = extract_equation_name(target_log)
        if not dataset:
            dataset = extract_dataset_name(target_log)
        candidates = extract_candidates_from_file(target_log)
        if not candidates:
            print(f"[verify error] Failed to extract any formula or code snippets from log: {target_log}")
            sys.exit(1)

    # Mode 2: Python file (--py)
    elif args.py:
        if not Path(args.py).is_file():
            print(f"Error: Python file does not exist: {args.py}")
            sys.exit(1)
        if not eq_name:
            eq_name = extract_equation_name(args.py)
        if not dataset:
            dataset = extract_dataset_name(args.py)
        candidates = [Path(args.py).read_text(encoding="utf-8")]

    # Mode 3: Direct String (--str)
    elif args.str:
        if not eq_name:
            print("Error: --eq-name MUST be provided when using --str")
            sys.exit(1)
        candidates = [args.str]

    if not dataset:
        dataset = "llm-srbench"

    result = verify_candidates(eq_name, candidates, llm_config, verbose=True, dataset=dataset)
    if result.startswith("[verify error]"):
        print(result)
        sys.exit(1)


if __name__ == "__main__":
    main()
