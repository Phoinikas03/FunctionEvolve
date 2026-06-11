"""
Selector: evolution tree parent selection strategy.

Responsibilities:
- Observe the evolution tree summary, make comprehensive judgments, and select multiple parent nodes for the current round
"""

from __future__ import annotations

import json
import math
import random
import re
import time
from typing import Any, Dict, List, Optional

import json_repair

from .llm_client import (
    add_guided_json_schema,
    build_openai_client,
    chat_completion,
    resolve_completion_max_tokens,
    with_json_only_instruction,
)


def _extract_code_block(raw: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    return match.group(1) if match else raw


def _is_dimension_only_context(context_prompt: str) -> bool:
    text = (context_prompt or "").lower()
    return (
        "aifeynman:" in text
        or "no natural-language domain background" in text
        or "no natural-language background" in text
    )


# ------------------------------------------------------------------ #
# Selector Prompts
# ------------------------------------------------------------------ #

SELECTOR_SYSTEM_PROMPT = """\
You are a strategic planning expert for symbolic regression search.

You will see a summary table of all generated formulas in the current evolution tree, each record contains:
- id          : Numeric unique identifier of the formula
- formula     : Formula in SymPy format
- description : Natural language description of the structure (including parameter behavior analysis)
- train_nmse  : Training set NMSE = MSE/Var(Y) (None means not yet evaluated; lower is better, <1 is better than mean prediction)
- n_children  : Number of offspring this node has produced
- n_params    : Number of tunable parameters
- depth       : AST tree depth (nesting level; higher means deeper function nesting)
- n_operators : Total number of operator nodes in AST tree (higher means longer expression)
- fitted_params : Fitted parameter values (scientific notation)

Your task is to select **{candidate_num} parent nodes** for the next evolution iteration.
The system will automatically perform AST-based structural mutations (delete subtrees, add new terms, etc.) on each parent to generate multiple candidate offspring.

Please consider the following when selecting:
1. **Structural simplicity first**: Do not simply select the nodes with the lowest NMSE. Prefer structurally simple nodes (fewer n_params, shallower depth, fewer n_operators) that already have relatively low NMSE (e.g., ~1e-3 order) — \
simple good formulas are more likely to produce meaningful improvements through mutation, while low NMSE in complex formulas often comes from overfitting
2. **Algebraic structure diversity**: The selected {candidate_num} parents should cover different algebraic structures as much as possible (e.g., polynomial, power law, exponential, trigonometric, rational, etc.), \
avoiding highly similar formulas. Diverse starting points enable more comprehensive exploration of the search space
3. **NMSE performance**: When structures are similar, prefer nodes with lower NMSE that still have room for improvement
4. **Number of offspring (n_children)**: If a node already has many offspring (e.g., n_children >= 5), \
it means that direction has been sufficiently explored; prefer nodes with fewer offspring
5. **Historical selection records**: Refer to the historical parent selection records. \
If certain formula structures have been repeatedly selected as parents in recent rounds but NMSE has not significantly decreased, they should be temporarily shelved in favor of other directions; \
if certain structures have not been explored recently, encourage selecting them as parents to expand the search space

Return only a JSON array with exactly {candidate_num} objects. Use the numeric `id` shown in the node summary as `parent_id`; do not copy the formula string into `parent_id`.

Required JSON schema:
{
  "type": "array",
  "minItems": {candidate_num},
  "maxItems": {candidate_num},
  "items": {
    "type": "object",
    "required": ["parent_id", "rationale"],
    "additionalProperties": false,
    "properties": {
      "parent_id": {"type": "integer"},
      "rationale": {"type": "string"}
    }
  }
}

Example item format (shape only; your actual output must contain exactly {candidate_num} objects):
[
  {"parent_id": 12, "rationale": "<selection rationale 1>"},
  {"parent_id": 7, "rationale": "<selection rationale 2>"}
]

- rationale: short final reason for selecting this parent.
"""

SELECTOR_USER_TEMPLATE = """\
Current evolution tree node summary ({n_nodes} nodes total, sorted by train_nmse ascending):

{tree_summary}

## Historical Parent Selection Records
{selection_history}

Please comprehensively analyze the above nodes and historical selection records, select {candidate_num} structurally diverse and concise parents, and output a JSON array."""

SELECTOR_SYSTEM_PROMPT_NO_AST = """\
You are a strategic planning expert for symbolic regression search.

You will see a summary of formulas in the current evolution tree. Each record contains:
- id          : Numeric unique identifier of the formula
- train_nmse  : Training set NMSE = MSE/Var(Y) (None means not yet evaluated; lower is better, <1 is better than mean prediction)
- n_children  : Number of offspring this node has produced
- fitted_params : Fitted parameter values when available (scientific notation)

(AST-derived metrics such as depth, operator counts, parameter counts, and natural-language structure descriptions are intentionally omitted.)

Your task is to select **{candidate_num} parent nodes** for the next evolution iteration.
The system will automatically perform structural mutations on each parent to generate candidate offspring.

Please consider the following when selecting:
1. **NMSE vs exploration**: Do not only pick the lowest train_nmse nodes; balance exploitation with exploration — moderately good nodes may still yield improvements through mutation
2. **Formula diversity from text alone**: The selected {candidate_num} parents should differ as much as possible judging only from the formula strings; avoid near-duplicate expressions and aim for visibly different functional forms in the text
3. **Number of offspring (n_children)**: If a node already has many offspring (e.g., n_children >= 5), that direction may be saturated; prefer nodes with fewer offspring when other factors are similar
4. **Historical selection records**: If certain formulas were repeatedly chosen as parents recently without NMSE improving, favor other ids; encourage ids that have not been explored recently

Return only a JSON array with exactly {candidate_num} objects. Use the numeric `id` shown in the node summary as `parent_id`; do not copy the formula string into `parent_id`.

Required JSON schema:
{
  "type": "array",
  "minItems": {candidate_num},
  "maxItems": {candidate_num},
  "items": {
    "type": "object",
    "required": ["parent_id", "rationale"],
    "additionalProperties": false,
    "properties": {
      "parent_id": {"type": "integer"},
      "rationale": {"type": "string"}
    }
  }
}

Example item format (shape only; your actual output must contain exactly {candidate_num} objects):
[
  {"parent_id": 12, "rationale": "<selection rationale 1>"},
  {"parent_id": 7, "rationale": "<selection rationale 2>"}
]

- rationale: short final reason for selecting this parent.
"""

SELECTOR_USER_TEMPLATE_NO_AST = """\
Current evolution tree node summary ({n_nodes} nodes total, sorted by train_nmse ascending):

{tree_summary}

## Historical Parent Selection Records
{selection_history}

Please analyze the above nodes and historical records (without relying on AST metadata that was hidden), select {candidate_num} diverse parents, and output a JSON array."""


class SelectorLLMAgent:
    """Strategic planner. Observes the entire evolution tree, makes comprehensive judgments, and selects multiple parent nodes."""

    def __init__(
        self,
        api_client: Any,
        model: str = "gpt-4o",
        max_retries: int = 3,
        retry_delay: float = 2.0,
        temperature: float = 0.4,
        max_tokens: int = 256,
        usage_logger: Any = None,
        anthropic_version: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        strip_ast_fields: bool = False,
    ):
        self.api = api_client
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._usage_logger = usage_logger
        self.anthropic_version = anthropic_version
        self.reasoning_effort = reasoning_effort
        self.strip_ast_fields = strip_ast_fields
        self._fallback_rng = random.Random(42)

    def plan(
        self,
        tree_summary: List[Dict[str, Any]],
        context_prompt: str = "",
        candidate_num: int = 5,
        selection_history: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Select multiple parent nodes based on the evolution tree summary."""
        actual_num = min(candidate_num, len(tree_summary))
        if len(tree_summary) < 2:
            return self._fallback_plan(tree_summary, actual_num)

        if self.strip_ast_fields:
            system_prompt = SELECTOR_SYSTEM_PROMPT_NO_AST.replace(
                "{candidate_num}", str(actual_num)
            )
            summary_str = self._format_tree_summary_no_ast(tree_summary)
            tmpl = SELECTOR_USER_TEMPLATE_NO_AST
        else:
            system_prompt = SELECTOR_SYSTEM_PROMPT.replace(
                "{candidate_num}", str(actual_num)
            )
            summary_str = self._format_tree_summary(tree_summary)
            tmpl = SELECTOR_USER_TEMPLATE
        if context_prompt:
            system_prompt += f"\n\n## Current Regression Task Background\n{context_prompt}"
            if _is_dimension_only_context(context_prompt):
                system_prompt += (
                    "\n\nFor this task, prioritize formula quality, structural diversity, "
                    "variable-name cues, and dimensional metadata when selecting parents."
                )

        history_str = self._format_selection_history(selection_history)
        user_prompt = tmpl.format(
            n_nodes=len(tree_summary),
            tree_summary=summary_str,
            candidate_num=actual_num,
            selection_history=history_str,
        )

        id_to_formula = {
            str(item["id"]): item["formula"]
            for item in tree_summary
            if item.get("id") is not None and item.get("formula")
        }
        id_to_value = {
            str(item["id"]): item["id"]
            for item in tree_summary
            if item.get("id") is not None and item.get("formula")
        }

        accumulated: List[Dict[str, Any]] = []
        seen_selector_ids = set()

        def _add_selection(item: Dict[str, Any]) -> bool:
            selector_id = str(item.get("parent_id", ""))
            if selector_id not in id_to_formula:
                return False
            if selector_id in seen_selector_ids:
                return False
            seen_selector_ids.add(selector_id)
            accumulated.append({
                "parent_id": id_to_value[selector_id],
                "parent_formula": id_to_formula[selector_id],
                "selector_id": selector_id,
                "rationale": item.get("rationale", ""),
            })
            return True

        for attempt in range(self.max_retries):
            try:
                raw = self._call_api(
                    system_prompt,
                    user_prompt,
                    json_schema=self._selection_schema(actual_num),
                )
                parsed_list = self._parse_json_list_response(raw)

                added = 0
                for item in parsed_list:
                    if _add_selection(item):
                        added += 1

                if not accumulated:
                    raise ValueError("No valid parent_id parsed")

                if len(accumulated) >= actual_num:
                    return accumulated[:actual_num]

                if attempt < self.max_retries - 1:
                    print(
                        f"  [Selector] Attempt {attempt+1} accumulated "
                        f"{len(accumulated)}/{actual_num} unique valid parents "
                        f"({added} new), retrying..."
                    )
                else:
                    print(
                        f"  [Selector] Accumulated "
                        f"{len(accumulated)}/{actual_num} unique valid parents "
                        "after retries."
                    )

            except Exception as e:
                print(f"  [Selector] Attempt {attempt+1} failed: {e}")
            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        fill = self._boltzmann_fill(
            tree_summary,
            actual_num - len(accumulated),
            exclude_selector_ids=seen_selector_ids,
        )
        if len(accumulated) < actual_num:
            print(
                f"  [Selector] Used {len(fill)} fallback parent suggestions "
                f"after retries ({len(accumulated) + len(fill)}/{actual_num} total)."
            )
        return (accumulated + fill)[:actual_num]

    def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        component: str = "selector",
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> str:
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
            json_schema or self._selection_schema(),
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
    def _selection_schema(candidate_num: Optional[int] = None) -> Dict[str, Any]:
        schema: Dict[str, Any] = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "parent_id": {"type": "integer"},
                    "rationale": {"type": "string"},
                },
                "required": ["parent_id", "rationale"],
                "additionalProperties": False,
            },
        }
        if candidate_num is not None:
            schema["minItems"] = candidate_num
            schema["maxItems"] = candidate_num
        return schema

    def _parse_json_list_response(self, raw: str) -> List[Dict[str, Any]]:
        text = _extract_code_block(raw)
        result = json_repair.loads(text)
        if isinstance(result, dict):
            return [result]
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
        raise ValueError(f"Failed to parse Selector response: {raw[:200]}")

    @staticmethod
    def _format_tree_summary(summary: List[Dict[str, Any]]) -> str:
        lines = []
        for item in summary:
            train_nmse = item.get("train_nmse")
            train_str = f"{train_nmse:.2e}" if train_nmse is not None else "Not evaluated"
            n_params = item.get("n_params", 0)
            fitted = item.get("fitted_params", "")
            fitted_line = f"\n    Fitted params: {fitted}" if fitted else ""
            desc = item.get("description", "")
            desc_line = f"    Description: {desc}\n" if desc else ""
            depth = item.get("depth", 0)
            n_ops = item.get("n_operators", 0)
            lines.append(
                f"  id: {item['id']}\n"
                f"    Formula: {item.get('formula', item.get('id', '?'))}\n"
                f"{desc_line}"
                f"    NMSE: train={train_str}\n"
                f"    n_params={n_params}  depth={depth}  n_operators={n_ops}"
                f"  |  n_children: {item.get('n_children', 0)}"
                + fitted_line
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_tree_summary_no_ast(summary: List[Dict[str, Any]]) -> str:
        """Same pool as full summary, but omit description / n_params / depth / n_operators."""
        lines = []
        for item in summary:
            train_nmse = item.get("train_nmse")
            train_str = f"{train_nmse:.2e}" if train_nmse is not None else "Not evaluated"
            fitted = item.get("fitted_params", "")
            fitted_line = f"\n    Fitted params: {fitted}" if fitted else ""
            lines.append(
                f"  id: {item['id']}\n"
                f"    Formula: {item.get('formula', item.get('id', '?'))}\n"
                f"    NMSE: train={train_str}"
                f"  |  n_children: {item.get('n_children', 0)}"
                + fitted_line
            )
        return "\n\n".join(lines)

    @staticmethod
    def _format_selection_history(history: Optional[List[List[str]]]) -> str:
        if not history:
            return "(First round, no history)"
        lines = []
        for i, parents in enumerate(history, 1):
            parents_str = ", ".join(parents) if parents else "None"
            lines.append(f"  Round {i}: {parents_str}")
        return "\n".join(lines)

    def _fallback_plan(
        self, summary: List[Dict[str, Any]], n: int = 1
    ) -> List[Dict[str, Any]]:
        results = self._boltzmann_fill(summary, n, exclude_selector_ids=set())
        return results or [
            {
                "parent_id": None,
                "parent_formula": "",
                "rationale": "Tree is empty",
            }
        ]

    def _boltzmann_fill(
        self,
        summary: List[Dict[str, Any]],
        n: int,
        exclude_selector_ids: set,
    ) -> List[Dict[str, Any]]:
        if n <= 0 or not summary:
            return []

        pool = [
            item for item in summary
            if str(item.get("id")) not in exclude_selector_ids
        ]
        if not pool:
            return []

        evaluated = [item for item in pool if item.get("train_nmse") is not None]
        pool = evaluated if evaluated else pool
        pool_sorted = sorted(
            pool,
            key=lambda item: (
                item.get("train_nmse")
                if item.get("train_nmse") is not None
                else float("inf")
            ),
        )

        m = len(pool_sorted)
        tau = max(self.temperature, 1e-9)
        log_weights = [-(rank / m) / tau for rank in range(m)]
        max_lw = max(log_weights)
        weights = [math.exp(lw - max_lw) for lw in log_weights]

        selected: List[Dict[str, Any]] = []
        remaining = list(range(m))
        remaining_w = list(weights)
        for _ in range(min(n, m)):
            total = sum(remaining_w)
            if total <= 0:
                break
            r = self._fallback_rng.random() * total
            cumsum = 0.0
            chosen_local = 0
            for j, w in enumerate(remaining_w):
                cumsum += w
                if cumsum >= r:
                    chosen_local = j
                    break
            chosen_idx = remaining.pop(chosen_local)
            remaining_w.pop(chosen_local)
            selected.append(pool_sorted[chosen_idx])

        def _fmt_nmse(v):
            if v is None:
                return "?"
            try:
                return f"{v:.3e}"
            except (TypeError, ValueError):
                return str(v)

        return [
            {
                "parent_id": item["id"],
                "parent_formula": item.get("formula", item["id"]),
                "selector_id": str(item["id"]),
                "rationale": (
                    "Boltzmann fallback fill after insufficient valid LLM selections: "
                    f"rank={pool_sorted.index(item)+1}/{m}, "
                    f"train_nmse={_fmt_nmse(item.get('train_nmse'))}"
                ),
            }
            for item in selected
        ]


# ------------------------------------------------------------------ #
# Mock Selector
# ------------------------------------------------------------------ #

class MockSelector:
    """Mock Selector: selects parents using Boltzmann rank-based sampling.

    Converts train NMSE to rank (lower NMSE = higher rank), then uses softmax(-rank / temperature)
    as sampling weights, so higher-quality nodes are more likely to be selected while
    lower-quality nodes still have a chance.
    """

    def __init__(self, variables: List[str], seed: int = 42,
                 temperature: float = 1.0):
        self.variables = variables
        self.temperature = temperature
        import random
        self._rng = random.Random(seed)

    def plan(
        self,
        tree_summary: List[Dict[str, Any]],
        context_prompt: str = "",
        candidate_num: int = 5,
        selection_history: Optional[List[List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        if not tree_summary:
            return [
                {
                    "parent_id": None,
                    "parent_formula": "",
                    "rationale": "Tree is empty",
                }
            ]
        evaluated = [n for n in tree_summary if n.get("train_nmse") is not None]
        pool = evaluated if evaluated else tree_summary
        k = min(candidate_num, len(pool))

        pool_sorted = sorted(
            pool,
            key=lambda n: (n.get("train_nmse") if n.get("train_nmse") is not None
                           else float("inf")),
        )

        import math
        n = len(pool_sorted)
        tau = max(self.temperature, 1e-9)
        log_weights = [-(rank / n) / tau for rank in range(n)]
        max_lw = max(log_weights)
        weights = [math.exp(lw - max_lw) for lw in log_weights]

        selected: List[Dict[str, Any]] = []
        remaining = list(range(n))
        remaining_w = list(weights)
        for _ in range(k):
            if not remaining:
                break
            total = sum(remaining_w)
            r = self._rng.random() * total
            cumsum = 0.0
            chosen_local = 0
            for j, w in enumerate(remaining_w):
                cumsum += w
                if cumsum >= r:
                    chosen_local = j
                    break
            chosen_idx = remaining[chosen_local]
            selected.append(pool_sorted[chosen_idx])
            remaining.pop(chosen_local)
            remaining_w.pop(chosen_local)

        def _fmt_nmse(v):
            if v is None:
                return "?"
            try:
                return f"{v:.3e}"
            except (TypeError, ValueError):
                return str(v)

        return [
            {
                "parent_id": p["id"],
                "parent_formula": p.get("formula", p["id"]),
                "selector_id": str(p["id"]),
                "rationale": (
                    f"Mock-Boltzmann: rank={pool_sorted.index(p)+1}/{n}, "
                    f"train_nmse={_fmt_nmse(p.get('train_nmse'))}"
                ),
            }
            for p in selected
        ]


# ------------------------------------------------------------------ #
# Factory function
# ------------------------------------------------------------------ #

def create_selector(
    model: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.4,
    max_tokens: int = 256,
    max_retries: int = 3,
    usage_logger: Any = None,
    llm_mode: str = "openai",
    anthropic_version: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    strip_ast_fields: bool = False,
) -> SelectorLLMAgent:
    """Create a Selector LLM agent."""
    client = build_openai_client(model, base_url, mode=llm_mode, api_key=api_key)
    return SelectorLLMAgent(
        api_client=client,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
        usage_logger=usage_logger,
        anthropic_version=anthropic_version,
        reasoning_effort=reasoning_effort,
        strip_ast_fields=strip_ast_fields,
    )
