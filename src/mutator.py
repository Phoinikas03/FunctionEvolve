"""
Mutation engine: AST rule-based mutation + LLM-driven mutation.

1. ASTMutator: Deterministic AST mutation (deletion + addition) with normalized outputs
2. LLMMutator: LLM-based heuristic mutation suggestions (ADDITION / SUBSTITUTION)
3. MockMutator: Test mutator that does not call LLM
"""

from __future__ import annotations

import json
import re
import time
from itertools import permutations
from typing import Any, Dict, List, Optional, Set

import sympy as sp
import json_repair

from .llm_client import (
    add_guided_json_schema,
    chat_completion,
    resolve_completion_max_tokens,
    with_json_only_instruction,
)
from .normalization import ExpressionNormalizer


class ASTMutator:
    """AST-based structural mutator."""

    def __init__(self, feature_names: List[str], overfit_min_depth: Optional[int] = None):
        self.feature_names = feature_names
        if overfit_min_depth is not None:
            self.OVERFIT_MIN_DEPTH = overfit_min_depth
        self.normalizer = ExpressionNormalizer(feature_names)

    # ------------------------------------------------------------------ #
    # AST Visualization
    # ------------------------------------------------------------------ #

    def get_labeled_ast(self, expr_str: str) -> str:
        """Generate indented AST tree text with sub-expressions on each internal node."""
        try:
            expr = sp.sympify(expr_str, locals=self.normalizer.make_locals(expr_str))
        except Exception:
            return f"(parse failed: {expr_str})"

        lines: List[str] = []
        self._fmt_node(expr, lines, "", True)
        return "\n".join(lines)

    def _fmt_node(self, node, lines: List[str], prefix: str, is_last: bool):
        conn = "└─ " if is_last else "├─ "
        ext = "   " if is_last else "│  "

        if node.is_Symbol or node.is_Number:
            lines.append(f"{prefix}{conn}{node}")
            return

        name = type(node).__name__
        friendly = {"Add": "+", "Mul": "×", "Pow": "^"}
        display = friendly.get(name, name)

        expr_str = str(node)
        if len(expr_str) > 60:
            expr_str = expr_str[:57] + "..."
        lines.append(f"{prefix}{conn}{display} [{expr_str}]")

        for i, arg in enumerate(node.args):
            self._fmt_node(arg, lines, prefix + ext, i == len(node.args) - 1)

    # ------------------------------------------------------------------ #
    # Deletion Mutations
    # ------------------------------------------------------------------ #

    def enumerate_deletions(self, expr_str: str) -> List[Dict[str, Any]]:
        """
        Enumerate all valid single-step deletion mutations.

        Includes:
        - Removing a term from Add
        - Removing a non-numeric factor from Mul
        - Unwrapping function wrappers (exp/log/sin/cos -> inner expression)
        - Removing exponent (x**n -> x)
        - Recursively applying the above in subtrees

        Output has normalized parameter names and is deduplicated by structural fingerprint.
        """
        try:
            expr = sp.sympify(expr_str, locals=self.normalizer.make_locals(expr_str))
        except Exception:
            return []

        raw: List[Dict[str, Any]] = []
        self._find_deletions(expr, raw)

        seen_keys: Set[str] = set()
        seen_keys.add(self.normalizer.struct_hash(expr))
        results = []
        feature_syms = {sp.Symbol(v) for v in self.feature_names}

        for c in raw:
            e = c["expr"]
            if e.is_number:
                continue
            if not (e.free_symbols & feature_syms):
                continue

            key = self.normalizer.struct_hash(e)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            results.append({
                "expression": str(e),
                "params": self.normalizer.collect_params(e),
                "mutation": f"[Deletion] {c['description']}",
            })
        return results

    def _find_deletions(self, expr, out: List[Dict]):
        if expr.func == sp.Add and len(expr.args) >= 2:
            for i, arg in enumerate(expr.args):
                rest = [a for j, a in enumerate(expr.args) if j != i]
                new = sp.Add(*rest) if len(rest) > 1 else rest[0]
                out.append({"expr": new, "description": f"Remove additive term {arg}"})

        if expr.func == sp.Mul and len(expr.args) >= 2:
            for i, arg in enumerate(expr.args):
                if arg.is_number:
                    continue
                rest = [a for j, a in enumerate(expr.args) if j != i]
                new = sp.Mul(*rest) if len(rest) > 1 else rest[0]
                out.append({"expr": new, "description": f"Remove multiplicative factor {arg}"})

        if expr.func in (sp.exp, sp.log, sp.sin, sp.cos, sp.tan, sp.sqrt):
            out.append({
                "expr": expr.args[0],
                "description": f"Unwrap {expr.func.__name__}() -> {expr.args[0]}",
            })

        if expr.func == sp.Pow:
            base, exp = expr.args
            if exp.is_number and exp != 1 and exp != 0:
                out.append({
                    "expr": base,
                    "description": f"Remove exponent {expr} -> {base}",
                })

        for i, arg in enumerate(expr.args):
            sub: List[Dict] = []
            self._find_deletions(arg, sub)
            for sc in sub:
                args = list(expr.args)
                args[i] = sc["expr"]
                try:
                    new = expr.func(*args)
                    out.append({"expr": new, "description": sc["description"]})
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Addition Mutations (term / factor / saturating divide / wrap)
    # ------------------------------------------------------------------ #

    def enumerate_additions(self, expr_str: str) -> List[Dict[str, Any]]:
        """
        Enumerate single-step structure-adding mutations on the parent f.

        The addition family attaches elementary content g while preserving f:
        additive terms (``f + g``), multiplicative factors (``f * g``),
        saturating divides (``f / (1 + g_s)`` for a denominator-safe subset),
        and unary wraps (``phi(c*f)`` / ``f**c``).

        Output has fresh parameter names and is deduplicated by expression string.
        See docs/mutator/architecture.md for the operator/content tables.
        """
        try:
            expr = sp.sympify(expr_str, locals=self.normalizer.make_locals(expr_str))
        except Exception:
            return []
        if expr.is_number:
            return []

        existing = self.normalizer.collect_params(expr)
        idx = max((int(p[1:]) for p in existing), default=-1) + 1
        base = f"({expr_str})"
        vars_ = self.feature_names
        var_pairs = list(permutations(vars_, 2))[:6] if len(vars_) >= 2 else []
        raw: List[Dict[str, Any]] = []

        def emit(expression: str, params: List[str], mutation: str) -> None:
            raw.append({"expression": expression, "params": params, "mutation": mutation})

        def fresh(n: int) -> List[str]:
            nonlocal idx
            params = [f"c{idx + i}" for i in range(n)]
            idx += n
            return params

        def emit_content(operator: str) -> None:
            """Attach the shared elementary content library as f+g or f*g."""
            op_symbol = "+" if operator == "term" else "*"
            op_desc = "term" if operator == "term" else "factor"

            def attach(g: str) -> str:
                return f"{base} + {g}" if operator == "term" else f"{base} * ({g})"

            for var in vars_:
                c1, c2 = fresh(2)
                g = f"{c1} * {var} + {c2}"
                emit(attach(g), existing + [c1, c2],
                     f"[Add] linear {op_desc} {op_symbol}{g}")

            for var in vars_:
                c1, c2, c3 = fresh(3)
                g = f"{c1} * sin({c2} * {var} + {c3})"
                emit(attach(g), existing + [c1, c2, c3],
                     f"[Add] sinusoidal {op_desc} {op_symbol}{g}")

            for var in vars_:
                c1, c2 = fresh(2)
                g = f"{c1} * {var}**{c2}"
                emit(attach(g), existing + [c1, c2],
                     f"[Add] power {op_desc} {op_symbol}{g}")

            for var in vars_:
                c1, c2 = fresh(2)
                g = f"{c1} * exp({c2} * {var})"
                emit(attach(g), existing + [c1, c2],
                     f"[Add] exponential {op_desc} {op_symbol}{g}")

            for var in vars_:
                c1, c2 = fresh(2)
                g = f"{c1} * log(1 + {c2} * {var})"
                emit(attach(g), existing + [c1, c2],
                     f"[Add] logarithmic {op_desc} {op_symbol}{g}")

            for var in vars_:
                c1, c2, c3, c4 = fresh(4)
                g = f"({c1} * {var} + {c2}) / ({c3} * {var} + {c4})"
                emit(attach(g), existing + [c1, c2, c3, c4],
                     f"[Add] rational {op_desc} {op_symbol}{g}")

            for v1, v2 in var_pairs:
                c1, c2, c3, c4 = fresh(4)
                g = f"({c1} * {v1} + {c2}) / ({c3} * {v2} + {c4})"
                emit(attach(g), existing + [c1, c2, c3, c4],
                     f"[Add] pair rational {op_desc} {op_symbol}{g}")

            for v1, v2 in var_pairs:
                c1, c2, c3 = fresh(3)
                g = f"{c1} * ({v1} + {c2})**{c3} * {v2}"
                emit(attach(g), existing + [c1, c2, c3],
                     f"[Add] power-law coupling {op_desc} {op_symbol}{g}")

        # ============================================================ #
        # Shared elementary content library: f + g and f * g
        # ============================================================ #
        emit_content("term")
        emit_content("factor")

        # ============================================================ #
        # Saturating divides: f / (1 + g_s), with a safe denominator subset
        # ============================================================ #
        for var in vars_:
            c1 = fresh(1)[0]
            emit(f"{base} / (1 + {c1} * {var})", existing + [c1],
                 f"[Add] saturating linear divide /(1+{c1}*{var})")

            c1, c2 = fresh(2)
            emit(f"{base} / (1 + {c1} * {var}**{c2})", existing + [c1, c2],
                 f"[Add] saturating power divide /(1+{c1}*{var}**{c2})")

            c1, c2 = fresh(2)
            emit(f"{base} / (1 + {c1} * exp({c2} * {var}))", existing + [c1, c2],
                 f"[Add] saturating exponential divide /(1+{c1}*exp({c2}*{var}))")

            c1, c2 = fresh(2)
            emit(f"{base} / (1 + {c1} * log(1 + {c2} * {var}))",
                 existing + [c1, c2],
                 f"[Add] saturating logarithmic divide /(1+{c1}*log(1+{c2}*{var}))")

        # ============================================================ #
        # Unary wraps: phi(c * f) and the generic power wrap f**c
        # ============================================================ #
        for fn in ("exp", "sin", "Abs"):
            c1 = f"c{idx}"
            emit(f"{fn}({c1} * {base})", existing + [c1], f"[Add] {fn} wrap {fn}({c1}*f)")
            idx += 1
        c1 = f"c{idx}"
        emit(f"log(1 + {c1} * {base})", existing + [c1], f"[Add] log wrap log(1+{c1}*f)")
        idx += 1
        c1 = f"c{idx}"
        emit(f"{base}**{c1}", existing + [c1], f"[Add] power wrap (f)**{c1}")
        idx += 1

        # ============================================================ #
        # Deduplication
        # ============================================================ #
        seen_exprs: Set[str] = set()
        results: List[Dict[str, Any]] = []
        for r in raw:
            if r["expression"] in seen_exprs:
                continue
            seen_exprs.add(r["expression"])
            results.append(r)
        return results


# ================================================================== #
# LLM-Driven Mutation
# ================================================================== #

# Operator-toolbox mutation prompt (LLM-driven structural edits)

MUTATION_SYSTEM_PROMPT = """\
You are a symbolic-regression expert. Propose **{target_num}** diverse structural **edits** to the \
parent formula so that each resulting candidate may fit the data better. Every edit takes the parent \
expression and transforms it into one new candidate expression.

## Structural Prior (read carefully)
Do **NOT** assume the ground-truth formula is a sum of terms. It may equally be:
- a single compact **monomial / product / ratio / composition** -- typical of first-principles physical \
laws, e.g. `c0*m1*m2/r**2`, `c0*exp(-c1*x**2)`, `c0/sqrt(1 - x**2/c1**2)`; or
- a **sum of several mechanistic terms** -- typical of empirical / dynamical models, e.g. \
`c0*(c1 - P/c2)*P + c3*P**c4`.
Let the parent's current fit guide you: if the parent already captures the trend (small residual), \
prefer small local refinements; if the parent looks structurally wrong, prefer a rewrite. Do **not** \
blindly keep growing the formula additively.

## Two Kinds of Edit (the program already enumerates the simple template edits -- do NOT repeat them)
The deterministic engine already covers simple template grafts (linear / power / exp / log / sin \
terms and factors, rational fractions, safe saturating divides, and single wraps) **and** all \
prunes / deletions / simplifications. See "Auto-Generated Candidates (do not repeat)" in the task. Your \
job is the part it cannot template:

- **ADDITION** (keep all of `f`; attach new material): `f + g`, `f * g`, or wrap `f` inside an outer \
function. Propose **out-of-library / domain-semantic** content -- e.g. `+ c3*exp(-c4/T)` (Arrhenius), \
`* 1/sqrt(1 - (c3*v)**2)` (Lorentz factor), cross-variable couplings `* x**c3 * y**c4`, or nested \
compositions `+ c3*log(1 + exp(c4*x))`. The whole parent is preserved.
- **SUBSTITUTION** (discard a subtree, replace it): use the annotated AST to locate a structurally wrong \
subtree and replace it with a different structure (`t -> s`, where `t` no longer appears). Use this when \
the parent's shape is mismatched -- not when you merely want to add a correction.

Do **not** propose pure simplifications / deletions -- the program handles those. Weight your \
{target_num} proposals toward whichever kind the parent's fit and structure call for; there is no fixed \
quota.

Elementary building blocks: power `x**c`, trigonometric `sin`/`cos`, exponential `exp`, logarithmic \
`log(1+x)`, and `sqrt`.

## Non-Elementary Function Usage
The following non-elementary functions are very common in scientific modeling and may be used with any \
operator above:
- **Abs(x)** (absolute value): modulus operations, amplitude extraction of sign-alternating variables. \
E.g. Abs(sin(x)) extracts oscillation amplitude, Abs(x)**c builds a V-shaped power law, log(1 + Abs(x)) \
symmetrizes logarithmic growth. Prefer Abs when a variable may be positive or negative but only the \
magnitude matters physically.
- **Max(x, 0)** (positive part / ReLU): threshold activation, one-sided response. E.g. Max(x - c, 0) \
responds only above threshold c. Suitable for systems with critical points / activation thresholds.
- Syntax: use `Abs(x)` and `Max(x, 0)` in SymPy.

## Using {knowledge_basis}
**Design structures informed by the knowledge below.** Different disciplines have typical formula priors.
{domain_knowledge}
You may introduce a structure from the knowledge above through either kind of edit -- as an **ADDITION** \
(a new factor / term / wrap that preserves `f`) or as a **SUBSTITUTION** (rewrite a mismatched subtree). \
When an edit is motivated by this knowledge, state the basis in the `mutation` field.

## Constant Notation Convention
Constant parameters should always appear in **multiplicative form**, never alone in a denominator:
- Correct: `c0 * x`, `c0 * sin(c1 * x)`, `c0 / (x + c1)`
- Incorrect: `x / c0`, `sin(x / c0)` -- a constant in the denominator is numerically unstable \
(approaching 0 causes overflow). To express a "scaled variable", write `c * var`, not `var / c`.

## Parameter Naming
Any **newly introduced** constant must use a **fresh** `cK` that does not appear in the parent formula \
or its fitted parameters; keep the names of parent constants you retain. E.g. if the parent uses \
`c0, c1, c2`, new constants start at `c3`.

## Task Background
{context}

## Output Format (strict compliance required)
Return only a JSON array of exactly {target_num} items.

Required JSON schema:
{{
  "type": "array",
  "items": {{
    "type": "object",
    "required": ["expression", "params", "mutation"],
    "additionalProperties": false,
    "properties": {{
      "expression": {{"type": "string"}},
      "params": {{"type": "array", "items": {{"type": "string"}}}},
      "mutation": {{"type": "string"}}
    }}
  }}
}}

Example item format:
[{{"expression": "<SymPy-parseable formula>", "params": ["c0", "c1"], "mutation": "ADD: <what changed and why>"}}]

- expression: use c0, c1, c2... for constants to be optimized; variables must come from the available variable list.
- params: list all parameter placeholders; if no constants, use [].
- mutation: **must** start with `ADD:` or `SUBST:`, followed by a brief rationale. When the edit is \
knowledge-driven, name the formula / law / structure it references.
- All functions must use SymPy/Python syntax: exp(x), log(x), sqrt(x), sin(x), cos(x), Abs(x), Max(x, 0).

## Available Variables
{variables}
"""


MUTATION_SYSTEM_PROMPT_NO_AST = (
    MUTATION_SYSTEM_PROMPT
    .replace(
        "discard a subtree, replace it",
        "discard a subexpression, replace it",
    )
    .replace(
        "use the annotated AST to locate a structurally wrong subtree",
        "use the parent formula and fitted behavior to identify a structurally wrong subexpression",
    )
)


def _is_dimension_only_context(context_prompt: str) -> bool:
    text = (context_prompt or "").lower()
    return (
        "aifeynman:" in text
        or "no natural-language domain background" in text
        or "no natural-language background" in text
    )


def build_generator_system_prompt(
    context: str,
    domain_knowledge: str,
    variables: str,
    target_num: int = 20,
    strip_ast_prompt: bool = False,
) -> str:
    """Build the operator-toolbox mutation system prompt.

    One template serves both domain-knowledge tasks and dimension-only
    (AI-Feynman) tasks; only the knowledge-section heading differs, so there is
    no longer any post-hoc string surgery on the rendered prompt.
    """
    knowledge_basis = (
        "Dimensional Information, Variable Names & Generic Structures"
        if _is_dimension_only_context(context)
        else "Domain Knowledge"
    )
    prompt_template = MUTATION_SYSTEM_PROMPT_NO_AST if strip_ast_prompt else MUTATION_SYSTEM_PROMPT
    return prompt_template.format(
        context=context,
        domain_knowledge=domain_knowledge,
        variables=variables,
        target_num=target_num,
        knowledge_basis=knowledge_basis,
    )


# User templates: the AST variant renders the parent AST block; the NO_AST
# variant omits it. ``strip_ast_prompt`` selects between them. The trailing
# instruction is identical in spirit -- ask for a diverse operator mix, no
# fixed per-operator quota.
MUTATION_USER_TEMPLATE = """\
## Parent Formula
{parent_formula}

## AST Structure
{labeled_ast}

## Fitted Parameters
{parent_fitted_params}

## Auto-Generated Candidates (do not repeat)
{auto_mutations_summary}

## Historical Best Formulas (top {topk})
{top_exprs}

Following the system instructions, propose {target_num} edits of two kinds: **ADDITION** (out-of-library / \
domain-semantic terms, factors, or wraps that preserve the parent) and **SUBSTITUTION** (rewrite a \
structurally wrong subtree located via the AST). Do not repeat the auto-generated template candidates \
above, and do not propose pure simplifications. There is no fixed quota -- weight the mix toward what \
the parent's AST and current fit suggest."""

MUTATION_USER_TEMPLATE_NO_AST = """\
## Parent Formula
{parent_formula}

## Fitted Parameters
{parent_fitted_params}

## Auto-Generated Candidates (do not repeat)
{auto_mutations_summary}

## Historical Best Formulas (top {topk})
{top_exprs}

Following the system instructions, propose {target_num} edits of two kinds: **ADDITION** (out-of-library / \
domain-semantic terms, factors, or wraps that preserve the parent) and **SUBSTITUTION** (rewrite a \
structurally wrong subtree). Do not repeat the auto-generated template candidates above, and do not \
propose pure simplifications. There is no fixed quota -- weight the mix toward what the parent's current \
fit suggests."""


def _extract_code_block(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    return match.group(1) if match else raw


class LLMMutator:
    """LLM-driven mutation suggester."""

    TARGET_MUTATION_NUM = 20

    def __init__(
        self,
        api_client: Any,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.8,
        max_tokens: int = 512,
        domain_knowledge: str = "",
        usage_logger: Any = None,
        anthropic_version: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        strip_ast_prompt: bool = False,
    ):
        self.api = api_client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._usage_logger = usage_logger
        self._domain_knowledge = domain_knowledge
        self.anthropic_version = anthropic_version
        self.reasoning_effort = reasoning_effort
        self.strip_ast_prompt = strip_ast_prompt

    @property
    def domain_knowledge(self) -> str:
        return self._domain_knowledge

    @domain_knowledge.setter
    def domain_knowledge(self, value: str) -> None:
        self._domain_knowledge = value

    def suggest_mutations(
        self,
        context_prompt: str,
        parent_formula: str,
        labeled_ast: str,
        parent_fitted_params: str = "",
        auto_mutations: Optional[List[str]] = None,
        variables: Optional[List[str]] = None,
        top_exprs: Optional[List[Dict[str, Any]]] = None,
        topk: int = 3,
    ) -> List[Dict[str, Any]]:
        """Suggest structural-edit proposals across the operator toolbox.

        Optionally omits the AST-heavy block (``strip_ast_prompt``).
        """
        variables = variables or []
        target_num = self.TARGET_MUTATION_NUM
        system_prompt = build_generator_system_prompt(
            context=context_prompt,
            domain_knowledge=self._domain_knowledge,
            variables=", ".join(variables),
            target_num=target_num,
            strip_ast_prompt=self.strip_ast_prompt,
        )
        tmpl = MUTATION_USER_TEMPLATE_NO_AST if self.strip_ast_prompt else MUTATION_USER_TEMPLATE

        auto_summary = "(None)"
        if auto_mutations:
            auto_summary = "\n".join(f"- {m}" for m in auto_mutations[:20])

        top_exprs_str = self._format_top_exprs(top_exprs, topk)

        user_prompt = tmpl.format(
            parent_formula=parent_formula,
            labeled_ast=labeled_ast,
            parent_fitted_params=parent_fitted_params or "No constant parameters",
            auto_mutations_summary=auto_summary,
            topk=topk,
            top_exprs=top_exprs_str,
            target_num=target_num,
        )

        accumulated: List[Dict[str, Any]] = []
        seen_exprs = set()

        def _add_suggestion(suggestion: Dict[str, Any]) -> bool:
            expr_key = " ".join(
                suggestion.get("expression", "").strip().split()
            )
            if not expr_key or expr_key in seen_exprs:
                return False
            seen_exprs.add(expr_key)
            accumulated.append(suggestion)
            return True

        for attempt in range(self.max_retries):
            try:
                raw = self._call_api(system_prompt, user_prompt,
                                     component="mutation")
                parsed = self._parse_json_array_response(raw)
                valid = [
                    p for p in parsed
                    if self._validate_response(p, variables)
                    and p.get("mutation")
                ]
                for suggestion in valid:
                    _add_suggestion(suggestion)
                if len(accumulated) >= target_num:
                    return accumulated[:target_num]
                if attempt < self.max_retries - 1:
                    print(
                        f"  [LLMMutator] Attempt {attempt+1} accumulated "
                        f"{len(accumulated)}/{target_num} unique valid "
                        "suggestions, retrying..."
                    )
                else:
                    print(
                        f"  [LLMMutator] Accumulated "
                        f"{len(accumulated)}/{target_num} unique valid "
                        "suggestions after retries."
                    )
            except Exception as e:
                print(f"  [LLMMutator] Attempt {attempt+1} call failed: {e}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        if len(accumulated) < target_num:
            print(
                f"  [LLMMutator] Used 0 fallback mutation suggestions "
                f"after retries ({len(accumulated)}/{target_num} total)."
            )
        return accumulated[:target_num]

    def _call_api(self, system_prompt: str, user_prompt: str,
                  component: str = "mutator") -> str:
        import time as _t
        t0 = _t.monotonic()
        
        # ✅ Build kwargs with anthropic_version and reasoning_effort if available
        messages = with_json_only_instruction([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        create_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": resolve_completion_max_tokens(
                self.model, messages, self.max_tokens),
        }
        add_guided_json_schema(
            create_kwargs,
            self._mutation_schema(),
            self.api,
        )
        if self.anthropic_version:
            create_kwargs["anthropic_version"] = self.anthropic_version
        if self.reasoning_effort:
            create_kwargs["reasoning_effort"] = self.reasoning_effort
        
        response = chat_completion(self.api, create_kwargs, component=component)
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

    @staticmethod
    def _mutation_schema() -> Dict[str, Any]:
        return {
            "type": "array",
            "minItems": LLMMutator.TARGET_MUTATION_NUM,
            "maxItems": LLMMutator.TARGET_MUTATION_NUM,
            "items": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                    "params": {"type": "array", "items": {"type": "string"}},
                    "mutation": {"type": "string"},
                },
                "required": ["expression", "params", "mutation"],
                "additionalProperties": False,
            },
        }

    @staticmethod
    def _parse_json_array_response(raw: str) -> List[Dict[str, Any]]:
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
        mutation = parsed.get("mutation", "")
        if not isinstance(mutation, str) or not mutation.startswith(("ADD:", "SUBST:")):
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


class MockMutator:
    """Mock LLM mutator: does not call LLM, returns empty list only."""

    def __init__(self):
        self._domain_knowledge = ""

    @property
    def domain_knowledge(self) -> str:
        return self._domain_knowledge

    @domain_knowledge.setter
    def domain_knowledge(self, value: str) -> None:
        self._domain_knowledge = value

    def suggest_mutations(self, **kwargs) -> List[Dict[str, Any]]:
        return []
