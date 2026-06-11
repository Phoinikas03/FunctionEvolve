from collections import defaultdict
import time
from typing import Dict, Set, List, Optional, Tuple, Any

import sympy as sp


def _finite_or_none(value: float) -> Optional[float]:
    return value if value != float("inf") else None


def _float_or_inf(value: Optional[float]) -> float:
    return float("inf") if value is None else float(value)


def _snapshot_list(iterable, attempts: int = 8) -> List[Any]:
    """Copy a mutable container view while tolerating concurrent updates."""
    for attempt in range(attempts):
        try:
            return list(iterable)
        except RuntimeError:
            time.sleep(0.001 * (attempt + 1))
    while True:
        try:
            return list(iterable)
        except RuntimeError:
            time.sleep(0.01)


def _snapshot_items(mapping, attempts: int = 8) -> List[Tuple[Any, Any]]:
    """Copy mapping items while tolerating concurrent updates."""
    for attempt in range(attempts):
        try:
            return list(mapping.items())
        except RuntimeError:
            time.sleep(0.001 * (attempt + 1))
    while True:
        try:
            return list(mapping.items())
        except RuntimeError:
            time.sleep(0.01)


class EvolutionNode:
    """
    An immutable formula node in the evolution tree.
    Upon initialization, performs SymPy parsing, full AST feature extraction,
    and natural language structure description generation.
    """

    def __init__(
        self,
        skeleton_str: str,
        parent_id: Optional[str] = None,
        node_id: Optional[int] = None,
    ):
        self.node_id = node_id
        self.skeleton_str = skeleton_str
        self.parent_id = parent_id
        self.train_nmse: float = float("inf")
        self.test_nmse: float = float("inf")
        self.ood_test_nmse: float = float("inf")
        self.is_evaluated: bool = False

        self.ast_features: Set[str] = set()
        self.operator_counts: Dict[str, int] = defaultdict(int)
        self.sympy_expr: Optional[sp.Basic] = None
        self.parse_error: bool = False

        # Fitted parameter information
        self.param_names: List[str] = []
        self.fitted_params: List[float] = []

        self.canonical_key: Optional[str] = None
        # Search step at which this structure first entered the tree.
        # None until stamped by EvolutionTree.add_node; 0 = seed (before Step 1).
        self.created_step: Optional[int] = None
        self.degeneration_candidates: List[Dict[str, Any]] = []
        self.degenerated_children: List[Dict[str, Any]] = []
        self.degeneration_reasons: List[str] = []

        # Structure description: prefer LLM-generated description, rule-based description as fallback
        self._rule_description: str = ""
        self._llm_description: str = ""

        self._parse_and_extract()
        self._rule_description = self._build_description()

    @property
    def description(self) -> str:
        return self._llm_description or self._rule_description

    @description.setter
    def description(self, value: str) -> None:
        self._llm_description = value

    def _parse_and_extract(self) -> None:
        """Recursively traverse AST, extract topology features with depth and operator frequencies."""
        try:
            self.sympy_expr = sp.sympify(self.skeleton_str)
        except Exception:
            self.parse_error = True
            return

        def traverse(node: sp.Basic, depth: int = 0) -> None:
            func_class = node.func.__name__
            self.ast_features.add(f"{func_class}(Depth:{depth})")
            self.operator_counts[func_class] += 1

            if func_class == "Pow" and len(node.args) == 2:
                exp = node.args[1]
                if exp.is_number and exp < 0:
                    self.ast_features.add(f"Fraction(Depth:{depth})")
                    self.operator_counts["Fraction"] += 1
                if exp == sp.Rational(1, 2):
                    self.ast_features.add(f"Sqrt(Depth:{depth})")

            for arg in node.args:
                traverse(arg, depth + 1)

        traverse(self.sympy_expr)

    def _build_description(self) -> str:
        """Generate a human-readable standalone structure description from AST features for Selector LLM reference."""
        if self.parse_error or self.sympy_expr is None:
            return "Failed to parse expression"

        ops = set(self.operator_counts.keys())
        parts: List[str] = []

        # Top-level structure
        top_op: Optional[str] = None
        for feat in self.ast_features:
            if "(Depth:0)" in feat:
                top_op = feat.split("(Depth:0)")[0]
                break

        top_map = {
            "Add": "additive combination", "Mul": "product form", "Pow": "power function",
            "exp": "exponential form", "log": "logarithmic form", "sin": "sine form",
            "cos": "cosine form", "Symbol": "single variable", "Integer": "constant",
        }
        if top_op in top_map:
            parts.append(f"top-level {top_map[top_op]}")
        elif top_op:
            parts.append(f"top-level {top_op}")

        # Special functions
        special = [f for f in ["exp", "log", "sin", "cos", "tan"] if f in ops]
        if special:
            parts.append(f"contains transcendental functions {'/'.join(special)}")

        if "Fraction" in ops:
            parts.append("contains fractions")
        if "Sqrt" in ops:
            parts.append("contains square root")
        if "Pow" in ops and "Fraction" not in ops and "Sqrt" not in ops:
            parts.append("contains powers")

        add_count = self.operator_counts.get("Add", 0)
        if add_count > 0:
            parts.append(f"{add_count} additive terms")

        depth = self.tree_depth
        if depth <= 2:
            parts.append("simple structure")
        elif depth <= 4:
            parts.append("moderate structure")
        else:
            parts.append(f"complex structure (depth {depth})")

        return ", ".join(parts) if parts else "basic expression"

    @property
    def tree_depth(self) -> int:
        if not self.ast_features:
            return 0
        depths = []
        for feat in self.ast_features:
            if "(Depth:" in feat:
                try:
                    d = int(feat.split("Depth:")[1].rstrip(")"))
                    depths.append(d)
                except ValueError:
                    pass
        return max(depths) if depths else 0

    def get_fitted_params_str(self) -> str:
        """Format fitted parameters as a scientific notation string."""
        if not self.param_names or not self.fitted_params:
            return ""
        parts = []
        for name, val in zip(self.param_names, self.fitted_params):
            parts.append(f"{name}={val:.4e}")
        return ", ".join(parts)

    @property
    def is_degenerated(self) -> bool:
        """A node is degenerated once it has any degeneration candidate."""
        return bool(self.degeneration_candidates)

    def add_degeneration_candidate(
        self,
        child_formula: Optional[str],
        reasons: Optional[List[str]] = None,
        kind: str = "simplified",
    ) -> None:
        """Record a degeneration-produced candidate before quality gating."""
        entry = {
            "child": child_formula,
            "kind": kind,
            "reasons": list(reasons or []),
        }
        if entry not in self.degeneration_candidates:
            self.degeneration_candidates.append(entry)

    def mark_degenerated(
        self,
        child_formula: Optional[str],
        reasons: Optional[List[str]] = None,
        kind: str = "simplified",
        child_train_nmse: Optional[float] = None,
        child_test_nmse: Optional[float] = None,
        child_ood_test_nmse: Optional[float] = None,
    ) -> None:
        """Hide this node after a degeneration outcome passes its gate."""
        self.add_degeneration_candidate(child_formula, reasons, kind=kind)
        entry = {
            "child": child_formula,
            "kind": kind,
            "reasons": list(reasons or []),
            "child_train_nmse": child_train_nmse,
            "child_test_nmse": child_test_nmse,
            "child_ood_test_nmse": child_ood_test_nmse,
        }
        if entry not in self.degenerated_children:
            self.degenerated_children.append(entry)
        if reasons:
            self.degeneration_reasons = list(reasons)

    def to_summary_dict(
        self,
        n_children: int = 0,
        parent_node_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Return a summary dict for Selector LLM, including parameter values and structural complexity.
        Only exposes train_nmse to avoid test/ood information leaking into search decisions."""
        result = {
            "id": self.node_id,
            "formula": self.skeleton_str,
            "train_nmse": round(self.train_nmse, 6) if self.is_evaluated and self.train_nmse < 1e9 else None,
            "parent_id": parent_node_id,
            "parent_formula": self.parent_id,
            "n_children": n_children,
            "n_params": len(self.param_names),
            "depth": self.tree_depth,
            "n_operators": sum(self.operator_counts.values()),
        }
        if self.description:
            result["description"] = self.description
        params_str = self.get_fitted_params_str()
        if params_str:
            result["fitted_params"] = params_str
        return result

    def to_state(self) -> Dict[str, Any]:
        """Serialize durable node state for checkpointing."""
        return {
            "skeleton_str": self.skeleton_str,
            "node_id": self.node_id,
            "parent_id": self.parent_id,
            "train_nmse": _finite_or_none(self.train_nmse),
            "test_nmse": _finite_or_none(self.test_nmse),
            "ood_test_nmse": _finite_or_none(self.ood_test_nmse),
            "is_evaluated": self.is_evaluated,
            "param_names": list(self.param_names),
            "fitted_params": list(self.fitted_params),
            "is_degenerated": self.is_degenerated,
            "canonical_key": self.canonical_key,
            "created_step": self.created_step,
            "degeneration_candidates": list(self.degeneration_candidates),
            "degenerated_children": list(self.degenerated_children),
            "degeneration_reasons": list(self.degeneration_reasons),
            "llm_description": self._llm_description,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "EvolutionNode":
        """Rebuild a node from checkpoint state.

        SymPy-derived fields are intentionally regenerated from the formula so
        checkpoints stay independent of Python object internals.
        """
        node = cls(
            str(state["skeleton_str"]),
            parent_id=state.get("parent_id"),
            node_id=state.get("node_id"),
        )
        node.train_nmse = _float_or_inf(state.get("train_nmse"))
        node.test_nmse = _float_or_inf(state.get("test_nmse"))
        node.ood_test_nmse = _float_or_inf(state.get("ood_test_nmse"))
        node.is_evaluated = bool(state.get("is_evaluated", False))
        node.param_names = list(state.get("param_names", []))
        node.fitted_params = list(state.get("fitted_params", []))
        node.canonical_key = state.get("canonical_key")
        node.created_step = state.get("created_step")
        node.degeneration_candidates = list(
            state.get("degeneration_candidates", []))
        node.degenerated_children = list(
            state.get("degenerated_children", []))
        node.degeneration_reasons = list(state.get("degeneration_reasons", []))
        if bool(state.get("is_degenerated", False)) and not node.degeneration_candidates:
            child_formula = state.get("degenerated_by")
            kind = "legacy"
            reasons = node.degeneration_reasons
            if node.degenerated_children:
                first_child = node.degenerated_children[0]
                child_formula = first_child.get("child")
                kind = first_child.get("kind", kind)
                reasons = first_child.get("reasons", reasons)
            node.add_degeneration_candidate(child_formula, reasons, kind=kind)
        node._llm_description = str(state.get("llm_description", ""))
        return node

    def __repr__(self) -> str:
        train_str = f"{self.train_nmse:.4f}" if self.train_nmse != float("inf") else "∞"
        return f"EvolutionNode(expr={self.skeleton_str!r}, train_nmse={train_str})"


class EvolutionTree:
    """
    Directed evolution tree. Each node is a symbolic formula.
    Supports multiple roots (initial seeds are all root nodes); each expansion
    attaches child nodes under the selected parent node.

    Core purposes:
    - Observation context for Selector LLM (get_tree_summary)
    - Information source for Generator LLM feedback (get_structural_diff_nl)
    - Maintains the globally best formula (best_node)
    """

    def __init__(self):
        self._nodes: Dict[str, EvolutionNode] = {}
        # parent_formula → [child_formulas]
        self._children: Dict[str, List[str]] = defaultdict(list)
        self._roots: List[str] = []
        self._next_node_id: int = 0
        # Current search step, updated by the searcher each iteration so that
        # add_node can stamp newly-created nodes with their birth round.
        # 0 means seed-initialization phase (before Step 1).
        self.current_step: int = 0

    # ------------------------------------------------------------------ #
    # Node operations
    # ------------------------------------------------------------------ #

    def add_node(
        self, formula: str, parent_formula: Optional[str] = None
    ) -> EvolutionNode:
        """Register a new node and attach it under the parent. Returns cached node if it already exists."""
        if formula not in self._nodes:
            node = EvolutionNode(
                formula,
                parent_id=parent_formula,
                node_id=self._next_node_id,
            )
            # Stamp the round at which this structure first appeared.
            node.created_step = self.current_step
            self._next_node_id += 1
            self._nodes[formula] = node
            if parent_formula and parent_formula in self._nodes:
                self._children[parent_formula].append(formula)
            else:
                if formula not in self._roots:
                    self._roots.append(formula)
        elif parent_formula:
            self.attach_child(parent_formula, formula)
        return self._nodes[formula]

    def _is_parent_chain_ancestor(
        self,
        maybe_ancestor: str,
        formula: str,
    ) -> bool:
        current = self._nodes.get(formula)
        seen = set()
        while current is not None and current.parent_id:
            if current.parent_id == maybe_ancestor:
                return True
            if current.parent_id in seen:
                return False
            seen.add(current.parent_id)
            current = self._nodes.get(current.parent_id)
        return False

    def attach_child(self, parent_formula: str, child_formula: str) -> bool:
        """Attach an existing child under a parent without changing ownership.

        This records duplicate/simplified relationships that should be visible
        to mature-node hiding, while avoiding trivial cycles.
        """
        if (
            not parent_formula
            or parent_formula == child_formula
            or parent_formula not in self._nodes
            or child_formula not in self._nodes
        ):
            return False
        if self._is_parent_chain_ancestor(child_formula, parent_formula):
            return False
        children = self._children[parent_formula]
        if child_formula in children:
            return False
        children.append(child_formula)
        return True

    def update_score(
        self,
        formula: str,
        train_nmse: float,
        test_nmse: float = float("inf"),
        ood_test_nmse: float = float("inf"),
        param_names: Optional[List[str]] = None,
        fitted_params: Optional[List[float]] = None,
    ) -> None:
        if formula in self._nodes:
            node = self._nodes[formula]
            node.train_nmse = train_nmse
            node.test_nmse = test_nmse
            node.ood_test_nmse = ood_test_nmse
            node.is_evaluated = True
            if param_names is not None:
                node.param_names = param_names
            if fitted_params is not None:
                node.fitted_params = fitted_params

    def get_node(self, formula: str) -> Optional[EvolutionNode]:
        return self._nodes.get(formula)

    def get_node_by_id(self, node_id: Any) -> Optional[EvolutionNode]:
        """Return a node by its numeric id while preserving formula-keyed storage."""
        try:
            target = int(node_id)
        except (TypeError, ValueError):
            return None
        for _, node in _snapshot_items(self._nodes):
            if node.node_id == target:
                return node
        return None

    def get_formula_by_id(self, node_id: Any) -> Optional[str]:
        node = self.get_node_by_id(node_id)
        return node.skeleton_str if node is not None else None

    def get_node_id(self, formula: Optional[str]) -> Optional[int]:
        if not formula:
            return None
        node = self._nodes.get(formula)
        return node.node_id if node is not None else None

    def resolve_formula_ref(self, ref: Any) -> Optional[str]:
        """Resolve either an old formula reference or a numeric node id to a formula."""
        if ref is None:
            return None
        ref_str = str(ref)
        if ref_str in self._nodes:
            return ref_str
        return self.get_formula_by_id(ref)

    def get_children(self, formula: str) -> List[str]:
        return list(self._children.get(formula, []))

    # ------------------------------------------------------------------ #
    # Selector context
    # ------------------------------------------------------------------ #

    def get_tree_summary(
        self,
        max_nodes: int = 20,
        exclude_formulas: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a list of node summaries for Selector LLM decision-making.
        Sorted: evaluated nodes by train_nmse ascending, unevaluated nodes appended at the end.
        ``exclude_formulas`` can omit dynamically collected sets such as mature
        nodes whose annealing budget is exhausted.
        Always excludes degenerated nodes.
        """
        exclude_formulas = set(exclude_formulas or set())
        nodes = [node for _, node in _snapshot_items(self._nodes)]
        evaluated = sorted(
            [n for n in nodes
             if n.is_evaluated
             and not n.is_degenerated
             and n.skeleton_str not in exclude_formulas],
            key=lambda n: n.train_nmse,
        )
        unevaluated = [n for n in nodes if not n.is_evaluated]
        ordered = (evaluated + unevaluated)[:max_nodes]
        return [
            n.to_summary_dict(
                n_children=len(self._children.get(n.skeleton_str, [])),
                parent_node_id=self.get_node_id(n.parent_id),
            )
            for n in ordered
        ]

    def get_structural_diff_nl(self, formula_a: str, formula_b: str) -> str:
        """Generate a natural language description of AST differences between two nodes for Generator feedback."""
        na = self._nodes.get(formula_a)
        nb = self._nodes.get(formula_b)
        if na is None or nb is None:
            return "Node does not exist"

        added = sorted(nb.ast_features - na.ast_features)[:5]
        removed = sorted(na.ast_features - nb.ast_features)[:5]

        parts = []
        if added:
            parts.append(f"Added: {', '.join(added)}")
        if removed:
            parts.append(f"Removed: {', '.join(removed)}")
        return "; ".join(parts) if parts else "Structurally similar (constants changed only)"

    # ------------------------------------------------------------------ #
    # Tree edit distance (auxiliary info for Selector diversity assessment)
    # ------------------------------------------------------------------ #

    def get_tree_edit_distance(self, formula_a: str, formula_b: str) -> int:
        """Compute the edit distance of post-order traversal sequences of two nodes (tree edit distance approximation)."""
        na = self._nodes.get(formula_a)
        nb = self._nodes.get(formula_b)
        if na is None or nb is None or na.parse_error or nb.parse_error:
            return -1

        def postorder(expr: sp.Basic) -> List[str]:
            labels: List[str] = []
            for arg in expr.args:
                labels.extend(postorder(arg))
            labels.append(expr.func.__name__)
            return labels

        seq_a = postorder(na.sympy_expr)
        seq_b = postorder(nb.sympy_expr)

        m, n = len(seq_a), len(seq_b)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, n + 1):
                temp = dp[j]
                cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = temp
        return dp[n]

    def get_pairwise_distances(
        self, formulas: List[str]
    ) -> Dict[Tuple[str, str], int]:
        """Compute pairwise tree edit distances for a given list of formulas."""
        distances: Dict[Tuple[str, str], int] = {}
        for i in range(len(formulas)):
            for j in range(i + 1, len(formulas)):
                d = self.get_tree_edit_distance(formulas[i], formulas[j])
                distances[(formulas[i], formulas[j])] = d
        return distances

    # ------------------------------------------------------------------ #
    # Global statistics
    # ------------------------------------------------------------------ #

    @property
    def best_node(self) -> Optional[EvolutionNode]:
        evaluated = [
            n for _, n in _snapshot_items(self._nodes)
            if n.is_evaluated and not n.is_degenerated
        ]
        return min(evaluated, key=lambda n: n.train_nmse) if evaluated else None

    @property
    def all_nodes(self) -> List[EvolutionNode]:
        return [node for _, node in _snapshot_items(self._nodes)]

    def get_stats(self) -> Dict[str, Any]:
        nodes = [node for _, node in _snapshot_items(self._nodes)]
        evaluated = [
            n for n in nodes
            if n.is_evaluated and not n.is_degenerated
        ]
        best = min(evaluated, key=lambda n: n.train_nmse) if evaluated else None
        return {
            "total_nodes": len(self._nodes),
            "evaluated": len(evaluated),
            "best_train_nmse": best.train_nmse if best else float("inf"),
            "best_test_nmse": best.test_nmse if best else float("inf"),
            "best_ood_test_nmse": best.ood_test_nmse if best else float("inf"),
            "best_expr": best.skeleton_str if best else None,
            "n_roots": len(self._roots),
        }

    def to_state(self, evaluated_only: bool = False) -> Dict[str, Any]:
        """Serialize durable tree state for checkpointing."""
        node_items = _snapshot_items(self._nodes)
        if evaluated_only:
            keep = {
                formula
                for formula, node in node_items
                if node.is_evaluated
            }
        else:
            keep = {formula for formula, _ in node_items}
        children_items = _snapshot_items(self._children)
        roots = _snapshot_list(self._roots)
        return {
            "nodes": [
                node.to_state()
                for formula, node in node_items
                if formula in keep
            ],
            "children": {
                parent: [
                    child for child in children
                    if child in keep
                ]
                for parent, children in children_items
                if parent in keep
            },
            "roots": [
                root for root in roots
                if root in keep
            ],
            "next_node_id": self._next_node_id,
        }

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "EvolutionTree":
        """Rebuild a tree from checkpoint state."""
        tree = cls()
        next_fallback_id = 0
        max_node_id = -1
        for node_state in state.get("nodes", []):
            if node_state.get("node_id") is None:
                node_state = dict(node_state)
                node_state["node_id"] = next_fallback_id
            node = EvolutionNode.from_state(node_state)
            tree._nodes[node.skeleton_str] = node
            if node.node_id is not None:
                max_node_id = max(max_node_id, int(node.node_id))
                next_fallback_id = max(next_fallback_id, int(node.node_id) + 1)
        tree._children = defaultdict(
            list,
            {
                str(parent): list(children)
                for parent, children in state.get("children", {}).items()
            },
        )
        tree._roots = list(state.get("roots", []))
        tree._next_node_id = max(
            int(state.get("next_node_id", 0) or 0),
            max_node_id + 1,
        )
        return tree

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"EvolutionTree(total={s['total_nodes']}, "
            f"evaluated={s['evaluated']}, "
            f"best_train={s['best_train_nmse']:.4f})"
        )


# Backward-compatible alias
GlobalASTTracker = EvolutionTree
