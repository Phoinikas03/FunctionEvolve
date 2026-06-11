"""
Symbolic regression framework entry point.

Search strategy overview
------------------------
Maintains an evolution tree where each node stores (formula, AST structure, structure description, NMSE).
Each iteration is collaboratively performed by multiple LLM modules:

  Selector LLM  : Observes the full tree summary and selects parent nodes
  ASTMutator     : Performs programmatic AST mutations on parents
  LLMMutator     : Suggests additional structural improvements
  Generator LLM  : Domain knowledge extraction, initial seed generation, formula description

LLM configuration for each component (model / base_url / temperature / total max_tokens)
is managed centrally via llm_config.yaml.

Usage examples
--------------
# Fully degenerated mode (no API needed, local functionality verification)
python main.py --degenerated-generator --degenerated-selector1 --degenerated-mutator1

# Using LLM services configured in llm_config.yaml
export OPENAI_API_KEY=sk-xxxx
python main.py --llm-config llm_config.yaml --dataset llm-srbench --split bio_pop_growth

# Mixed mode (degenerated Generator, LLM Selector + LLM Mutator)
python main.py --llm-config llm_config.yaml --degenerated-generator --dataset aifeynman --split feynmanequations
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Raise the GC generation-2 threshold dramatically.  The main process
# accumulates millions of SymPy Expr objects over a long run; the default
# threshold triggers stop-the-world gen-2 collections every ~70 k allocations,
# causing multi-minute pauses that stall the dispatcher and create CPU bubbles.
gc.set_threshold(100_000, 100, 100)

import numpy as np

from src.dataset import (
    SRDataset,
    dataset_split_names,
    find_equation_split,
    list_equations,
    normalize_dataset_name,
    resolve_requested_cases,
    set_repo_dir,
)
from src.evaluator import Evaluator
from src.evolution_tree import EvolutionTree
from src.generator import LLMGenerator, MockGenerator, create_generator
from src.selector import SelectorLLMAgent, MockSelector, create_selector
from src.mutator import LLMMutator, MockMutator
from src.llm_client import build_openai_client, LLMUsageLogger
from src.search import (
    TreeSearch,
    active_phase_snapshot,
    terminate_active_subprocesses,
)
from src.fair_executor import FairEvalExecutor
from src.checkpoint import load_checkpoint


_CODE_VERSION_CACHE: Optional[str] = None
_CODE_VERSION_LOCK = threading.Lock()


def _validate_log_component(label: str, value: Optional[str]) -> Optional[str]:
    """Return a path component after rejecting accidental nested paths."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text in {".", ".."} or "/" in text or "\\" in text:
        raise ValueError(
            f"{label} must be a single path component, got {value!r}. "
            "Use --dataset for the dataset layer and --run-tag only for the run layer."
        )
    return text


def _get_code_version() -> str:
    """Return a cheap, process-cached git version string for run logs."""
    global _CODE_VERSION_CACHE
    if _CODE_VERSION_CACHE is not None:
        return _CODE_VERSION_CACHE

    with _CODE_VERSION_LOCK:
        if _CODE_VERSION_CACHE is not None:
            return _CODE_VERSION_CACHE
        try:
            repo_dir = Path(__file__).parent
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            git_dirty = subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=str(repo_dir),
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).decode().strip()
            _CODE_VERSION_CACHE = f"{git_hash}{'+dirty' if git_dirty else ''}"
        except Exception:
            _CODE_VERSION_CACHE = "unknown"
        return _CODE_VERSION_CACHE


# ------------------------------------------------------------------ #
# Built-in demo data
# ------------------------------------------------------------------ #

def _make_demo_dataset() -> SRDataset:
    """Population exponential growth demo data. True formula: N0 * exp(0.3 * t)"""
    rng = np.random.default_rng(42)
    t = np.linspace(0.1, 10.0, 80)
    N0_vals = rng.uniform(5.0, 20.0, size=80)
    y = N0_vals * np.exp(0.3 * t) + rng.normal(0, 0.5, size=80)

    X_train = np.column_stack([t[:60], N0_vals[:60]])
    y_train = y[:60]
    X_test  = np.column_stack([t[60:], N0_vals[60:]])
    y_test  = y[60:]

    return SRDataset.from_arrays(
        X_train=X_train,
        y_train=y_train,
        symbols=["t", "N0"],
        X_test=X_test,
        y_test=y_test,
        symbol_descs=["Time", "Initial population size"],
        expression="N0 * exp(r * t)",
        equation_name="demo_pop_growth",
    )


# ------------------------------------------------------------------ #
# Ground Truth baseline evaluation
# ------------------------------------------------------------------ #

def _evaluate_ground_truth(
    ds: SRDataset, evaluator: Evaluator, log_fn
) -> Optional[Dict[str, float]]:
    """
    Evaluate the ground truth expression in the dataset, print NMSE as a search reference baseline.
    Automatically identifies symbols in the expression that are not feature variables, treating them as constants to fit.

    Returns
    -------
    dict with keys 'train', 'test', 'ood' mapping to NMSE floats,
    or None if evaluation failed.
    """
    import re
    import sympy as sp

    expr_str = ds.expression
    if not expr_str:
        return None

    # In srbench, ground truth often uses function notation like P(t); replace with bare variable name P
    all_known = set(ds.symbols) | set(ds.feature_names)
    for sym in all_known:
        expr_str = re.sub(rf'\b{re.escape(sym)}\s*\([^)]*\)', sym, expr_str)

    # Data format: 0.123_w means 0.123 * _w, where _w is an independent parameter to fit; insert multiplication sign
    expr_str = re.sub(r'(\d+\.?\d*)(_[A-Za-z]\w*)', r'\1*\2', expr_str)

    feature_set = set(ds.feature_names)
    try:
        # Pre-scan identifiers, register all as Symbols to prevent gamma/beta etc. from being parsed as SymPy built-in functions
        identifiers = set(re.findall(r'\b([A-Za-z_]\w*)\b', expr_str))
        sympy_funcs = {'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs',
                       'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh',
                       'sign', 'floor', 'ceiling', 'pi', 'E', 'I'}
        local_syms = {
            name: sp.Symbol(name)
            for name in identifiers if name not in sympy_funcs
        }
        sympy_expr = sp.sympify(expr_str, locals=local_syms)
        all_syms = {str(s) for s in sympy_expr.free_symbols}
        param_names = sorted(all_syms - feature_set)
    except Exception as e:
        log_fn(f"[Ground Truth] Parse failed: {e}")
        return None

    result = evaluator.evaluate_skeleton(sympy_expr, param_names)

    def _fmt(v: float) -> str:
        return f"{v:.2e}" if v < 1e9 and v != float("inf") else "—"

    lines = [f"[Ground Truth] Expression: {ds.expression}"]
    if expr_str != ds.expression:
        lines.append(f"[Ground Truth] Normalized: {expr_str}")
    if param_names:
        parts = [f"{n}={v:.4e}" for n, v in zip(param_names, result.best_params)]
        lines.append(f"[Ground Truth] Fitted params: {', '.join(parts)}")
    lines.append(
        f"[Ground Truth] train={_fmt(result.train_nmse)}"
        f"  test={_fmt(result.test_nmse)}"
        f"  ood={_fmt(result.ood_test_nmse)}"
    )
    lines.append("")
    for line in lines:
        log_fn(line)

    return {
        "train": result.train_nmse,
        "test": result.test_nmse,
        "ood": result.ood_test_nmse,
    }


# ------------------------------------------------------------------ #
# Main workflow
# ------------------------------------------------------------------ #

def run(
    dataset: Optional[str] = None,
    split: Optional[str] = None,
    equation: Optional[str] = None,
    llm_config: Optional[dict] = None,
    llm_config_path: Optional[str] = None,
    llm_max_retries: int = 5,
    verify_llm_config: Optional[dict] = None,
    verify_llm_config_path: Optional[str] = None,
    degenerated_generator: bool = False,
    degenerated_selector1: bool = False,
    degenerated_selector2: bool = False,
    degenerated_mutator1: bool = False,
    degenerated_mutator2: bool = False,
    degenerated_mutator3: bool = False,
    max_steps: int = 30,
    n_seeds: int = 20,
    selector_context_size: int = 1000,
    candidate_num: int = 5,
    optimizer: str = "Structure",
    max_mature_nodes: int = 5,
    mature_train_threshold: Optional[float] = None,
    mature_anneal_budget: int = 0,
    overfit_min_depth: int = 6,
    n_eval_workers: Optional[int] = None,
    timeout: float = 120.0,
    mutator_seen_topk: int = 100,
    max_params: int = 10,
    refine_output: bool = False,
    enable_describe: bool = False,
    verbose: bool = False,
    enable_preprocessing: bool = False,
    run_tag: Optional[str] = None,
    eval_executor: Optional[Any] = None,
    eval_owner_id: Optional[str] = None,
    resume: bool = False,
    checkpoint_dir: Optional[str] = None,
    shutdown_event: Optional[threading.Event] = None,
    code_version: Optional[str] = None,
) -> dict:
    """
    Assemble and launch evolution tree search.

    Parameters
    ----------
    dataset              : benchmark family (llm-srbench, llm-srbench-noise*, aifeynman)
    split                : split name within the benchmark family
    equation             : specific equation name within the split
    llm_config           : dict of per-component LLM parameters (loaded from YAML)
    llm_config_path      : llm_config YAML file path (used for log subdirectory naming)
    llm_max_retries      : max LLM call attempts per request (default 5)
    verify_llm_config    : optional LLM config used only for GT equivalence verification
    verify_llm_config_path : optional verify config YAML path (for logging/debug)
    degenerated_generator: True -> use hardcoded seeds + empty description, no LLM calls
    degenerated_selector1: True -> Boltzmann rank sampling for parents, no LLM calls
    degenerated_selector2: True -> LLM selector with AST/description fields stripped from prompts
    degenerated_mutator1 : True -> use only programmatic mutations (delete/add), no LLM calls
    degenerated_mutator2 : True -> use only LLM mutations, no programmatic mutations
    degenerated_mutator3 : True -> same as mutator2, plus LLM prompts without AST blocks or AST-local guidance
    max_steps            : maximum evolution steps
    n_seeds              : number of initial seeds
    selector_context_size : max number of nodes passed to Selector each time
    candidate_num        : number of parents selected by Selector per step
    optimizer            : constant optimizer name (Structure / DE / CMA-ES / L-BFGS-B / least_squares)
    max_mature_nodes     : early stop after collecting this many mature nodes (default 5)
    mature_train_threshold : mature node train NMSE threshold (None auto-sets to 1.05*GT)
    mature_anneal_budget : times a mature node may still be selected as parent (default 0)
    refine_output        : whether to perform final scan after search ends (default False)
    enable_describe      : whether to generate LLM structural descriptions for nodes (default False)
    run_tag              : experiment tag for isolating parallel runs (inserted into log path)
    eval_executor        : optional shared executor for candidate evaluation across equations
    eval_owner_id        : owner key used by shared eval executor scheduling
    code_version         : precomputed code version string for logs
    """
    if degenerated_mutator1 and degenerated_mutator2:
        raise ValueError(
            "degenerated_mutator1 (programmatic only) and degenerated_mutator2 (LLM only) cannot be enabled simultaneously."
        )
    if degenerated_mutator1 and degenerated_mutator3:
        raise ValueError(
            "degenerated_mutator1 (programmatic only) and degenerated_mutator3 (LLM, no AST prompts) cannot be enabled simultaneously."
        )
    if degenerated_mutator2 and degenerated_mutator3:
        raise ValueError(
            "degenerated_mutator2 and degenerated_mutator3 cannot be enabled simultaneously; use only degenerated_mutator3."
        )
    if degenerated_selector1 and degenerated_selector2:
        raise ValueError(
            "degenerated_selector1 and degenerated_selector2 cannot be enabled simultaneously."
        )

    # ---- 1. Dataset ----
    if dataset:
        try:
            dataset_name = normalize_dataset_name(dataset)
        except ValueError:
            # Backward-compatible programmatic API: run(dataset="matsci", equation=...)
            dataset_name = "llm-srbench"
            split = split or dataset
        if split is None:
            if equation:
                split = find_equation_split(equation, dataset_name=dataset_name)
                if split is None:
                    raise ValueError(
                        f"Equation '{equation}' not found in dataset '{dataset_name}'"
                    )
            else:
                split = dataset_split_names(dataset_name)[0]
        ds = SRDataset.from_benchmark(dataset_name, split, equation_name=equation)
        ds.load(quiet=not verbose)
    else:
        if verbose:
            print("No --dataset specified, using built-in demo dataset (population exponential growth).")
        ds = _make_demo_dataset()

    if verbose:
        print(ds.summary())
        print()

    # ---- Log file ----
    dataset_log_tag = dataset_name if dataset else "demo"
    run_tag = _validate_log_component("run_tag", run_tag)
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    date_str = now.strftime("%Y-%m-%d")
    eq_tag = ds.equation_name.replace("/", "_") if ds.equation_name else "unknown"
    log_root = Path(__file__).parent / "logs" / dataset_log_tag
    if run_tag:
        log_root = log_root / run_tag
    if llm_config_path:
        model_tag = Path(llm_config_path).stem
        log_root = log_root / model_tag
    else:
        model_tag = "default"
    ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else (log_root / "checkpoint")
    ckpt_path = ckpt_dir / f"{eq_tag}.json"
    diagnostic_log_path = ckpt_dir / f"{eq_tag}_diagnostics.log"
    resume_state = load_checkpoint(ckpt_path) if resume else None
    if resume_state and resume_state.get("status") == "completed":
        best = resume_state.get("best", {})
        nodes = (
            resume_state.get("search_state", {})
            .get("tree", {})
            .get("nodes", [])
        )
        best_node = None
        if nodes:
            best_node = min(
                (
                    n for n in nodes
                    if n.get("is_evaluated") and n.get("train_nmse") is not None
                ),
                key=lambda n: n.get("train_nmse", float("inf")),
                default=None,
            )
        if verbose:
            print(f"[checkpoint] {eq_tag} already completed: {ckpt_path}")
        return {
            "expression": (
                best.get("expression")
                or (best_node or {}).get("skeleton_str")
            ),
            "train_nmse": best.get(
                "train_nmse",
                (best_node or {}).get("train_nmse", float("inf")),
            ),
            "test_nmse": best.get(
                "test_nmse",
                (best_node or {}).get("test_nmse", float("inf")),
            ),
            "ood_test_nmse": best.get(
                "ood_test_nmse",
                (best_node or {}).get("ood_test_nmse", float("inf")),
            ),
            "params": (best_node or {}).get("param_names", []),
            "best_params": (best_node or {}).get("fitted_params", []),
            "description": (best_node or {}).get("llm_description", ""),
            "ast_features": [],
        }

    log_dir = log_root / date_str
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(log_dir / f"{eq_tag}_{timestamp}.txt")
    llm_usage_path = str(log_dir / f"{eq_tag}_llm_usage_{timestamp}.csv")
    if verbose:
        print(f"Log file: {log_path}")
        print(f"LLM usage: {llm_usage_path}")
        print(f"Checkpoint: {ckpt_path}")
        print(f"Diagnostics: {diagnostic_log_path}")

    feature_names = ds.feature_names

    # ---- 2. Evaluator (train / test / ood_test three-way evaluation) ----
    evaluator = Evaluator(
        feature_names=feature_names,
        X_train=ds.X_train,
        y_train=ds.y_train,
        X_test=ds.X_test if ds.X_test.size > 0 else None,
        y_test=ds.y_test if ds.y_test.size > 0 else None,
        X_ood_test=ds.X_ood_test if ds.X_ood_test.size > 0 else None,
        y_ood_test=ds.y_ood_test if ds.y_ood_test.size > 0 else None,
        optimizer=optimizer,
        timeout=timeout,
    )

    tree = EvolutionTree()

    # ---- 3. LLM agents ----
    needs_llm = (
        not degenerated_generator
        or not degenerated_selector1
        or not degenerated_mutator1
    )
    if needs_llm and not llm_config:
        raise ValueError(
            "Current config requires LLM (non-degenerated components exist in generator/selector/mutator),\n"
            "please specify a YAML config file via --llm-config."
        )

    usage_logger = None
    _cfg = llm_config or {}

    def _resolve(component: str, field: str, fallback=None):
        return _cfg.get(component, {}).get(field, fallback)

    # ✅ Validate reasoning_effort parameter compatibility
    def _validate_reasoning_config(component_name: str, cfg: dict):
        """
        验证 reasoning_effort 配置兼容性。
        GPT-5.2 使用 reasoning_effort 时有严格限制：
        - temperature 必须为 1.0
        - 不能使用 top_p, frequency_penalty, presence_penalty, top_k
        """
        if "reasoning_effort" not in cfg:
            return
        
        model = cfg.get("model", "")
        if "gpt-5" not in model.lower():
            return
        
        warnings = []
        
        # 检查 temperature
        if "temperature" in cfg and cfg["temperature"] != 1.0:
            warnings.append(
                f"⚠️  {component_name}: reasoning_effort={cfg['reasoning_effort']!r} 要求 temperature=1.0，"
                f"但配置中为 {cfg['temperature']}。系统会在运行时强制设置为 1.0"
            )
        
        # 检查不兼容参数
        incompatible_params = ["top_p", "frequency_penalty", "presence_penalty", "top_k"]
        found_incompatible = [p for p in incompatible_params if p in cfg]
        if found_incompatible:
            warnings.append(
                f"⚠️  {component_name}: reasoning_effort 不能与以下参数共存：{found_incompatible}。"
                f"系统会在运行时移除这些参数"
            )
        
        if warnings:
            for warn in warnings:
                print(f"[Config Warning] {warn}")
    
    if llm_config and needs_llm:
        usage_logger = LLMUsageLogger(llm_usage_path)
        
        # 验证所有组件的 reasoning_effort 配置
        for comp_name in ("generator", "domain_knowledge", "seed_generation", "describe_batch", "selector", "mutator"):
            if comp_name in _cfg:
                _validate_reasoning_config(comp_name, _cfg[comp_name])

    # --- generator ---
    gen_mode = _resolve("generator", "mode", "openai")
    gen_api_key = _resolve("generator", "api_key")

    if degenerated_generator:
        generator = MockGenerator(variables=feature_names)
        if verbose:
            print("  [Degenerated] Generator: hardcoded seeds + empty description")
    else:
        gen_model  = _resolve("generator", "model")
        gen_url    = _resolve("generator", "base_url")
        gen_temp   = _resolve("generator", "temperature", 0.8)
        gen_tokens = _resolve("generator", "max_tokens", 4096)
        gen_anthropic_version = _resolve("generator", "anthropic_version")
        gen_reasoning_effort = _resolve("generator", "reasoning_effort")
        if not gen_model:
            raise ValueError("generator.model must be specified in llm_config")

        _gen_sub_names = ("domain_knowledge", "seed_generation", "describe_batch")
        comp_overrides: Dict[str, Dict[str, Any]] = {}
        
        if gen_anthropic_version:
            comp_overrides["generator"] = {"anthropic_version": gen_anthropic_version}
        if gen_reasoning_effort:
            if "generator" not in comp_overrides:
                comp_overrides["generator"] = {}
            comp_overrides["generator"]["reasoning_effort"] = gen_reasoning_effort
        
        for sub in _gen_sub_names:
            sub_cfg = _cfg.get(sub, {})
            if not sub_cfg:
                continue
            ov: Dict[str, Any] = {}
            sub_model  = sub_cfg.get("model", gen_model)
            sub_url    = sub_cfg.get("base_url", gen_url)
            sub_mode = sub_cfg.get("mode", gen_mode)
            sub_api_key = sub_cfg.get("api_key", gen_api_key)
            if sub_model != gen_model or sub_url != gen_url or sub_mode != gen_mode or sub_api_key != gen_api_key:
                ov["api_client"] = build_openai_client(
                    sub_model, sub_url, mode=sub_mode, api_key=sub_api_key)
            if sub_model != gen_model:
                ov["model"] = sub_model
            if "temperature" in sub_cfg:
                ov["temperature"] = sub_cfg["temperature"]
            if "max_tokens" in sub_cfg:
                ov["max_tokens"] = sub_cfg["max_tokens"]
            if "anthropic_version" in sub_cfg:
                ov["anthropic_version"] = sub_cfg["anthropic_version"]
            elif (sub_model == gen_model and sub_url == gen_url and gen_anthropic_version):
                ov["anthropic_version"] = gen_anthropic_version
            if "reasoning_effort" in sub_cfg:
                ov["reasoning_effort"] = sub_cfg["reasoning_effort"]
            elif (sub_model == gen_model and sub_url == gen_url and gen_reasoning_effort):
                ov["reasoning_effort"] = gen_reasoning_effort
            if ov:
                comp_overrides[sub] = ov

        generator = create_generator(
            model=gen_model,
            base_url=gen_url, api_key=gen_api_key, temperature=gen_temp,
            max_tokens=gen_tokens, usage_logger=usage_logger,
            max_retries=llm_max_retries,
            component_overrides=comp_overrides or None,
            enable_preprocessing=enable_preprocessing,
            llm_mode=gen_mode,
        )

    # --- selector ---
    if degenerated_selector1:
        selector = MockSelector(variables=feature_names)
        if verbose:
            print("  [Degenerated] Selector1: Boltzmann rank sampling")
    else:
        sel_model  = _resolve("selector", "model", _resolve("generator", "model"))
        sel_url    = _resolve("selector", "base_url", _resolve("generator", "base_url"))
        sel_api_key = _resolve("selector", "api_key", gen_api_key)
        sel_temp   = _resolve("selector", "temperature", 0.4)
        sel_tokens = _resolve("selector", "max_tokens", 4096)
        sel_mode = _resolve("selector", "mode", gen_mode)
        sel_anthropic_version = _resolve("selector", "anthropic_version")
        sel_reasoning_effort = _resolve("selector", "reasoning_effort")
        selector = create_selector(
            model=sel_model,
            base_url=sel_url, api_key=sel_api_key, temperature=sel_temp,
            max_tokens=sel_tokens, usage_logger=usage_logger,
            max_retries=llm_max_retries,
            llm_mode=sel_mode,
            anthropic_version=sel_anthropic_version,
            reasoning_effort=sel_reasoning_effort,
            strip_ast_fields=degenerated_selector2,
        )
        if verbose and degenerated_selector2:
            print("  [Degenerated] Selector2: LLM selection without AST/description fields in prompt")

    # --- llm_mutator ---
    skip_programmatic = False

    def _build_llm_mutator(strip_ast_prompt: bool = False):
        mut_model  = _resolve("mutator", "model", _resolve("generator", "model"))
        mut_url    = _resolve("mutator", "base_url", _resolve("generator", "base_url"))
        mut_api_key = _resolve("mutator", "api_key", gen_api_key)
        mut_temp   = _resolve("mutator", "temperature", 0.8)
        mut_tokens = _resolve("mutator", "max_tokens", 4096)
        mut_mode = _resolve("mutator", "mode", gen_mode)
        mut_anthropic_version = _resolve("mutator", "anthropic_version")
        mut_reasoning_effort = _resolve("mutator", "reasoning_effort")
        client = build_openai_client(mut_model, mut_url, mode=mut_mode, api_key=mut_api_key)
        return LLMMutator(
            api_client=client, model=mut_model,
            temperature=mut_temp, max_tokens=mut_tokens,
            max_retries=llm_max_retries,
            usage_logger=usage_logger,
            anthropic_version=mut_anthropic_version,
            reasoning_effort=mut_reasoning_effort,
            strip_ast_prompt=strip_ast_prompt,
        )

    if degenerated_mutator1:
        llm_mutator = MockMutator()
        if verbose:
            print("  [Degenerated] Mutator: programmatic mutations only")
    elif degenerated_mutator3:
        skip_programmatic = True
        llm_mutator = _build_llm_mutator(strip_ast_prompt=True)
        if verbose:
            print(
                "  [Degenerated] Mutator: LLM only, prompts without AST blocks or AST-local guidance"
            )
    elif degenerated_mutator2:
        skip_programmatic = True
        llm_mutator = _build_llm_mutator()
        if verbose:
            print("  [Degenerated] Mutator: LLM mutations only")
    else:
        llm_mutator = _build_llm_mutator()

    if verbose and llm_config:
        _all_comps = ("generator", "domain_knowledge", "seed_generation",
                      "describe_batch", "selector", "mutator")
        for comp in _all_comps:
            cc = _cfg.get(comp, {})
            if cc:
                parts = [f"{k}={v}" for k, v in cc.items()]
                print(f"  [LLM Config] {comp}: {', '.join(parts)}")
        print()

    # ---- 4. Searcher ----
    searcher = TreeSearch(
        dataset=ds,
        evaluator=evaluator,
        tree=tree,
        selector=selector,
        generator=generator,
        llm_mutator=llm_mutator,
        skip_programmatic_mutations=skip_programmatic,
        max_steps=max_steps,
        n_seeds=n_seeds,
        selector_context_size=selector_context_size,
        candidate_num=candidate_num,
        max_mature_nodes=max_mature_nodes,
        mature_train_threshold=mature_train_threshold,
        mature_anneal_budget=mature_anneal_budget,
        overfit_min_depth=overfit_min_depth,
        n_parent_workers=candidate_num,
        n_eval_workers=n_eval_workers,
        timeout=timeout,
        mutator_seen_topk=mutator_seen_topk,
        max_params=max_params,
        refine_output=refine_output,
        enable_describe=enable_describe,
        verbose=verbose,
        log_path=log_path,
        eval_executor=eval_executor,
        eval_owner_id=eval_owner_id or eq_tag,
        checkpoint_path=str(ckpt_path),
        diagnostic_log_path=str(diagnostic_log_path),
        checkpoint_metadata={
            "run_tag": run_tag,
            "model_tag": model_tag,
            "dataset": dataset,
            "equation": ds.equation_name,
            "equation_tag": eq_tag,
            "log_path": log_path,
            "llm_usage_path": llm_usage_path,
            "diagnostic_log_path": str(diagnostic_log_path),
        },
        shutdown_event=shutdown_event,
    )

    # Record git version once per process.  Full worktree status is expensive
    # on GPFS with large log/checkpoint directories, especially per equation.
    searcher._log(f"Code version: {code_version or _get_code_version()}")
    _degen_flags = []
    if degenerated_generator:
        _degen_flags.append("generator")
    if degenerated_selector1:
        _degen_flags.append("selector1(Boltzmann)")
    if degenerated_selector2:
        _degen_flags.append("selector2(LLM no AST fields)")
    if degenerated_mutator1:
        _degen_flags.append("mutator1(programmatic)")
    if degenerated_mutator2:
        _degen_flags.append("mutator2(LLM-only)")
    if degenerated_mutator3:
        _degen_flags.append("mutator3(LLM-only, no AST in mutator prompt)")
    _degen_str = ", ".join(_degen_flags) if _degen_flags else "none"
    searcher._log(f"Run config: dataset={dataset}, split={split}, equation={equation}")
    searcher._log(f"optimizer={optimizer}, max_steps={max_steps}, n_seeds={n_seeds}")
    if eval_executor is not None:
        _global_workers = getattr(eval_executor, "max_workers", "unknown")
        searcher._log(
            f"global_workers={_global_workers}, eval_owner_id={searcher.eval_owner_id}"
        )
    is_aifeynman = ds.dataset_identifier.startswith("aifeynman:")
    is_noisy_srbench = ds.dataset_identifier.startswith("llm-srbench-noise")
    gt_multiplier = 1.5 if (is_aifeynman or is_noisy_srbench) else 1.05
    if is_aifeynman:
        searcher._log(f"sample_size={len(ds.y_train)}")
    searcher._log(f"GT_MULTIPLIER={gt_multiplier}")
    searcher._log(f"Degenerated components: {_degen_str}")
    if llm_config:
        for _comp_name in ("generator", "selector", "mutator"):
            _comp_cfg = llm_config.get(_comp_name, {})
            if _comp_cfg:
                _parts = [f"{k}={v}" for k, v in _comp_cfg.items()]
                searcher._log(f"  {_comp_name}: {', '.join(_parts)}")
    searcher._log(f"Log file: {log_path}")
    searcher._log(ds.summary())
    searcher._log("")

    # ---- Ground Truth baseline & auto threshold ----
    gt_nmse = _evaluate_ground_truth(ds, evaluator, searcher._log)

    if gt_nmse is not None:
        if mature_train_threshold is None:
            searcher.mature_train_threshold = gt_nmse["train"] * gt_multiplier
        searcher.mature_test_threshold = gt_nmse["test"] * gt_multiplier
        searcher._log(
            f"[Early stop threshold] Auto-set based on GT*{gt_multiplier}:"
            f" train<{searcher.mature_train_threshold:.2e}"
        )
    else:
        searcher._log(
            f"[Early stop threshold] No GT baseline, using default/manual values:"
            f" train<{searcher.mature_train_threshold:.2e}"
        )

    if resume_state:
        searcher.load_checkpoint_state(resume_state)

    # ---- 5. Initialize seeds + search ----
    searcher.initialize_seeds()
    searcher.run()

    # ---- 6. Output results ----
    result = searcher.get_best_result()

    def _fmt(v: float) -> str:
        return f"{v:.2e}" if v < 1e9 and v != float("inf") else "—"

    def _fmt_params(names, values):
        if not names or not values:
            return "no constant params"
        return ", ".join(f"{n}={v:.4e}" for n, v in zip(names, values))

    final_lines = [
        "",
        "=" * 60,
        "Final search results",
        "=" * 60,
        f"Best expression     : {result.get('expression')}",
        f"Structure desc      : {result.get('description')}",
        f"Fitted params       : {_fmt_params(result.get('params', []), result.get('best_params', []))}",
        f"Train NMSE          : {_fmt(result.get('train_nmse', float('inf')))}",
        f"Test NMSE           : {_fmt(result.get('test_nmse', float('inf')))}",
        f"OOD Test NMSE       : {_fmt(result.get('ood_test_nmse', float('inf')))}",
    ]
    for line in final_lines:
        if verbose:
            print(line)
        searcher._log(line)

    searcher.close_log()
    if usage_logger is not None:
        usage_logger.close()

    if shutdown_event is not None and shutdown_event.is_set():
        return result

    # ---- 7. GT equivalence verification ----
    verify_config = verify_llm_config or llm_config
    if verify_config:
        try:
            from verify import extract_equation_name, extract_candidates_from_file, verify_candidates, extract_dataset_name
            eq_name = extract_equation_name(log_path)
            dataset_name = dataset or extract_dataset_name(log_path)
            candidates = extract_candidates_from_file(log_path)
            if not candidates:
                if verbose:
                    print(f"[verify] No candidates extracted from log, skipping verification")
            else:
                if verbose and verify_llm_config_path:
                    print(f"[verify] Using separate LLM config: {verify_llm_config_path}")
                verify_text = verify_candidates(eq_name, candidates, verify_config, verbose=verbose, dataset=dataset_name)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(verify_text + "\n")
                if verbose:
                    print(f"[verify] Verification results appended to log: {log_path}")
        except Exception as e:
            if verbose:
                print(f"[verify error] GT verification failed: {e}")

    return result


# ------------------------------------------------------------------ #
# Concurrent test for entire split
# ------------------------------------------------------------------ #

def _find_completed_equations(
    log_dir: Path,
    model_tag: Optional[str] = None,
    run_tag: Optional[str] = None,
) -> Dict[str, Path]:
    """
    Recursively scan logs directory (including date subdirectories) to find equations with completed searches.

    Parameters
    ----------
    log_dir   : logs root directory
    model_tag : if specified, only scan logs in paths containing this model subdirectory
    run_tag   : if specified, only scan logs in paths containing this run_tag subdirectory

    Returns
    -------
    {eq_tag: log_path} mapping, where eq_tag is the equation name ('/' replaced with '_').
    """
    completed: Dict[str, Path] = {}
    if not log_dir.exists():
        return completed

    for log_file in sorted(log_dir.rglob("*.txt")):
        parts = log_file.parts
        if model_tag and model_tag not in parts:
            continue
        if run_tag and run_tag not in parts:
            continue
        eq_tag = log_file.stem.rsplit("_", 2)[0]
        if eq_tag in completed:
            continue
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
            if "Search completed" in text:
                completed[eq_tag] = log_file
        except Exception:
            continue
    return completed


def _default_checkpoint_dir(
    dataset_name: str,
    run_tag: Optional[str],
    llm_config_path: Optional[str],
    checkpoint_dir: Optional[str] = None,
) -> Path:
    """Return the checkpoint directory for a run configuration."""
    if checkpoint_dir:
        return Path(checkpoint_dir)
    run_tag = _validate_log_component("run_tag", run_tag)
    root = Path(__file__).parent / "logs" / normalize_dataset_name(dataset_name)
    if run_tag:
        root = root / run_tag
    if llm_config_path:
        root = root / Path(llm_config_path).stem
    return root / "checkpoint"


def _find_completed_checkpoints(checkpoint_dir: Path) -> Dict[str, Path]:
    """Find equations whose checkpoint status is completed."""
    completed: Dict[str, Path] = {}
    if not checkpoint_dir.exists():
        return completed
    for ckpt_file in sorted(checkpoint_dir.glob("*.json")):
        if ckpt_file.name == "manifest.json":
            continue
        try:
            state = load_checkpoint(ckpt_file) or {}
        except Exception:
            continue
        if state.get("status") != "completed":
            continue
        eq_tag = str(
            state.get("equation_tag")
            or state.get("equation")
            or ckpt_file.stem
        ).replace("/", "_")
        completed[eq_tag] = ckpt_file
    return completed


def run_split(
    split_name: str,
    dataset_name: str = "llm-srbench",
    equation_workers: int = 4,
    global_workers: int = 16,
    eval_owner_max_inflight: int = 32,
    eval_executor_log_interval: float = 10.0,
    dynamic_global_workers: bool = True,
    global_worker_initial: int = 1,
    global_worker_cpu_grow_threshold: float = 85.0,
    global_worker_cpu_shrink_threshold: float = 90.0,
    global_worker_cpu_check_interval: float = 0.1,
    **run_kwargs,
) -> Dict[str, dict]:
    """Run all pending equations in one split with a shared eval worker queue."""
    deprecated_global_workers = run_kwargs.pop("global_eval_workers", None)
    if deprecated_global_workers is not None and global_workers == 16:
        global_workers = int(deprecated_global_workers)
    nested = run_splits(
        [split_name],
        dataset_name=dataset_name,
        equation_workers=equation_workers,
        global_workers=global_workers,
        eval_owner_max_inflight=eval_owner_max_inflight,
        eval_executor_log_interval=eval_executor_log_interval,
        dynamic_global_workers=dynamic_global_workers,
        global_worker_initial=global_worker_initial,
        global_worker_cpu_grow_threshold=global_worker_cpu_grow_threshold,
        global_worker_cpu_shrink_threshold=global_worker_cpu_shrink_threshold,
        global_worker_cpu_check_interval=global_worker_cpu_check_interval,
        **run_kwargs,
    )
    return nested.get(split_name, {})


def _preload_llm_sdk(llm_config: Optional[dict]) -> None:
    """Import the OpenAI SDK before launching many worker threads."""
    if not llm_config:
        return

    for component in ("generator", "selector", "mutator"):
        cfg = llm_config.get(component, {})
        mode = cfg.get("mode", llm_config.get("generator", {}).get("mode", "openai"))
        if mode in ("openai", "vllm"):
            from openai import OpenAI  # noqa: F401
            print("[LLMClient] OpenAI SDK preloaded before worker threads")
            return


def _iter_completed_futures(futures, shutdown_event: threading.Event):
    """Yield completed futures, but stop waiting promptly after Ctrl-C."""
    pending = set(futures)
    while pending:
        if shutdown_event.is_set():
            for fut in pending:
                fut.cancel()
            break
        done, pending = concurrent.futures.wait(
            pending,
            timeout=0.5,
            return_when=concurrent.futures.FIRST_COMPLETED,
        )
        for fut in done:
            yield fut


def _fmt_counter(counter: Any, limit: int = 8) -> str:
    if not isinstance(counter, dict) or not counter:
        return "{}"
    items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
    body = ", ".join(f"{k}:{v}" for k, v in items[:limit])
    if len(items) > limit:
        body += f", ...(+{len(items) - limit})"
    return "{" + body + "}"


def _start_eval_executor_logger(
    eval_executor: FairEvalExecutor,
    *,
    shutdown_event: threading.Event,
    log_path: Path,
    interval: float = 10.0,
) -> tuple[threading.Event, threading.Thread]:
    """Log live shared-eval executor counters outside checkpoint throttling."""
    stop_event = threading.Event()
    interval = max(1.0, float(interval))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(line: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _line(reason: str = "tick") -> str:
        snap = eval_executor.snapshot()
        phases = active_phase_snapshot()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cpu_percent = snap.get("cpu_percent")
        cpu_display = (
            f"{float(cpu_percent):.1f}%"
            if isinstance(cpu_percent, (int, float))
            else "na"
        )
        return (
            f"[FairEvalExecutor {now} {reason}] "
            f"backend={snap.get('backend', 'thread')} "
            f"result_backend={snap.get('result_backend', 'shared-queue')} "
            f"workers={snap.get('max_workers')} "
            f"active_limit={snap.get('active_worker_limit', 'na')} "
            f"dynamic_workers={snap.get('dynamic_workers', 'na')} "
            f"cpu={cpu_display} "
            f"cpu_thresholds={float(snap.get('cpu_grow_threshold', 0.0)):.1f}/{float(snap.get('cpu_shrink_threshold', 0.0)):.1f} "
            f"cpu_monitor={snap.get('cpu_monitor_alive', 'na')} "
            f"alive_workers={snap.get('worker_processes_alive', 'na')} "
            f"owner_cap={snap.get('owner_max_inflight')} "
            f"owner_worker_affinity_limit={snap.get('owner_worker_affinity_limit', 'na')} "
            f"owners={snap.get('owners')} "
            f"ready_owners={snap.get('ready_owners')} "
            f"queued_total={snap.get('queued_total')} "
            f"submit_q={snap.get('submit_queue_size', 'na')} "
            f"dispatch_q={snap.get('dispatch_queue_size', 'na')} "
            f"scheduler_result_q={snap.get('scheduler_result_queue_size', 'na')} "
            f"running_total={snap.get('running_total')} "
            f"inflight_total={snap.get('inflight_total', 'na')} "
            f"idle_workers={snap.get('idle_workers', 'na')} "
            f"input_channels={snap.get('input_channels', 'na')} "
            f"idle_channels={snap.get('idle_worker_channels', 'na')} "
            f"busy_channels={snap.get('busy_worker_channels', 'na')} "
            f"oldest_queued={float(snap.get('oldest_queued_age_s', 0.0)):.1f}s "
            f"oldest_running={float(snap.get('oldest_running_age_s', 0.0)):.1f}s "
            f"submitted_total={snap.get('submitted_total')} "
            f"finished_total={snap.get('finished_total')} "
            f"worker_started={snap.get('worker_started')} "
            f"worker_finished={snap.get('worker_finished')} "
            f"dispatcher_alive={snap.get('dispatcher_alive', 'na')} "
            f"dispatchers={snap.get('dispatcher_threads_alive', snap.get('dispatchers', 'na'))}/{snap.get('dispatchers', 'na')} "
            f"dispatch_put_active={snap.get('dispatch_put_active_count', 'na')} "
            f"dispatch_put_ms={float(snap.get('dispatch_put_last_ms', 0.0)):.1f}/{float(snap.get('dispatch_put_max_ms', 0.0)):.1f} "
            f"dispatch_put_slow={snap.get('dispatch_put_slow_total', 'na')} "
            f"context_payload_sent={snap.get('context_payload_sent_total', 'na')} "
            f"context_cache_hit={snap.get('context_cache_hit_total', 'na')} "
            f"context_spillover={snap.get('context_spillover_total', 'na')} "
            f"dispatch_last_age={float(snap.get('dispatch_last_age_s', -1.0)):.1f}s "
            f"dispatch_task={snap.get('dispatch_task_active', '')!r} "
            f"dispatch_take_none={snap.get('dispatch_take_none_total', 'na')} "
            f"dispatch_cancelled={snap.get('dispatch_cancelled_before_run_total', 'na')} "
            f"dispatch_waits={snap.get('dispatch_wait_total', 'na')} "
            f"result_alive={snap.get('result_collector_alive', 'na')} "
            f"result_get={snap.get('result_loop_get_active', 'na')} "
            f"result_readers={snap.get('result_readers_alive', 'na')}/{snap.get('result_collector_threads', snap.get('max_workers', 'na'))} "
            f"result_readers_active={snap.get('result_readers_active', 'na')} "
            f"retire={snap.get('retire_threads_alive', 'na')}/{snap.get('retire_workers', 'na')} "
            f"retire_q={snap.get('retire_queue_size', 'na')} "
            f"retire_pressure={snap.get('retire_pressure', 'na')} "
            f"completion={snap.get('completion_threads_alive', 'na')}/{snap.get('completion_workers', 'na')} "
            f"completion_q={snap.get('completion_queue_size', 'na')} "
            f"retire_active={snap.get('retire_task_active_count', 'na')} "
            f"complete_active={snap.get('complete_task_active', 'na')} "
            f"complete_active_count={snap.get('complete_task_active_count', 'na')} "
            f"result_recv={snap.get('result_received_total', 'na')} "
            f"result_ingest_backlog={snap.get('result_ingest_backlog', 'na')} "
            f"result_retired={snap.get('result_retired_total', 'na')} "
            f"result_done={snap.get('result_completed_total', 'na')} "
            f"retire_backlog={snap.get('result_retire_backlog', 'na')} "
            f"future_backlog={snap.get('future_set_backlog', 'na')} "
            f"result_seen_age={float(snap.get('result_loop_last_seen_age_s', -1.0)):.1f}s "
            f"result_retired_age={float(snap.get('result_loop_last_retired_age_s', -1.0)):.1f}s "
            f"result_done_age={float(snap.get('result_loop_last_completed_age_s', -1.0)):.1f}s "
            f"retire_ms={float(snap.get('retire_task_last_ms', 0.0)):.1f}/{float(snap.get('retire_task_max_ms', 0.0)):.1f} "
            f"future_ms={float(snap.get('future_set_last_ms', 0.0)):.1f}/{float(snap.get('future_set_max_ms', 0.0)):.1f} "
            f"retire_slow={snap.get('retire_task_slow_total', 'na')} "
            f"future_slow={snap.get('future_set_slow_total', 'na')} "
            f"result_err={snap.get('result_loop_error', '')!r} "
            f"worker_exec={snap.get('worker_exec_started', 'na')}/{snap.get('worker_exec_finished', 'na')} "
            f"worker_put={snap.get('worker_put_started', 'na')}/{snap.get('worker_put_finished', 'na')} "
            f"worker_put_pending={snap.get('worker_put_pending', 'na')} "
            f"worker_put_failed={snap.get('worker_put_failed', 'na')} "
            f"queued_by_fn={_fmt_counter(snap.get('queued_by_fn'))} "
            f"running_by_fn={_fmt_counter(snap.get('running_by_fn'))} "
            f"running_by_owner={_fmt_counter(snap.get('running_by_owner'), limit=12)} "
            f"phases={_fmt_counter(phases, limit=16)}"
        )

    def _loop() -> None:
        _write(_line("start"))
        while not shutdown_event.is_set() and not stop_event.wait(interval):
            _write(_line())
        _write(_line("stop"))

    thread = threading.Thread(
        target=_loop,
        name="fair-eval-logger",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _start_equation_scheduler_logger(
    equation_states: Dict[str, Dict[str, Any]],
    state_lock: threading.Lock,
    *,
    shutdown_event: threading.Event,
    log_path: Path,
    equation_workers: int,
    interval: float = 10.0,
) -> tuple[threading.Event, threading.Thread]:
    """Log equation-thread supply so eval starvation is distinguishable."""
    stop_event = threading.Event()
    interval = max(1.0, float(interval))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(line: str) -> None:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def _line(reason: str = "tick") -> str:
        now_mono = time.monotonic()
        with state_lock:
            states = [dict(v) for v in equation_states.values()]

        by_state = Counter(str(s.get("state", "unknown")) for s in states)
        running = [s for s in states if s.get("state") == "running"]
        submitted = [s for s in states if s.get("state") == "submitted"]
        running_by_split = Counter(str(s.get("split", "")) for s in running)
        submitted_by_split = Counter(str(s.get("split", "")) for s in submitted)
        done_by_split = Counter(
            str(s.get("split", ""))
            for s in states
            if s.get("state") in {"done", "error"}
        )

        def _age(state: Dict[str, Any], key: str) -> float:
            value = state.get(key)
            if not isinstance(value, (int, float)):
                return 0.0
            return max(0.0, now_mono - float(value))

        oldest_running = max((_age(s, "started_at") for s in running), default=0.0)
        oldest_submitted = max(
            (_age(s, "submitted_at") for s in submitted), default=0.0)
        stale_running = sum(
            1 for s in running if _age(s, "updated_at") >= 120.0)
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"[EquationScheduler {now} {reason}] "
            f"workers={equation_workers} "
            f"total={len(states)} "
            f"submitted_waiting={by_state.get('submitted', 0)} "
            f"running={by_state.get('running', 0)} "
            f"done={by_state.get('done', 0)} "
            f"error={by_state.get('error', 0)} "
            f"oldest_submitted={oldest_submitted:.1f}s "
            f"oldest_running={oldest_running:.1f}s "
            f"stale_running={stale_running} "
            f"submitted_by_split={_fmt_counter(dict(submitted_by_split))} "
            f"running_by_split={_fmt_counter(dict(running_by_split))} "
            f"done_by_split={_fmt_counter(dict(done_by_split))}"
        )

    def _loop() -> None:
        _write(_line("start"))
        while not shutdown_event.is_set() and not stop_event.wait(interval):
            _write(_line())
        _write(_line("stop"))

    thread = threading.Thread(
        target=_loop,
        name="equation-scheduler-logger",
        daemon=True,
    )
    thread.start()
    return stop_event, thread


def _force_exit_process_group(exit_code: int = 1) -> None:
    """Terminate the whole process group, falling back to immediate exit."""
    try:
        os.killpg(os.getpgid(os.getpid()), signal.SIGKILL)
    except Exception:
        os._exit(exit_code)


def _start_shutdown_watchdog(
    shutdown_event: threading.Event,
    *,
    grace: float = 300.0,
    label: str = "shutdown",
) -> threading.Thread:
    """Force-exit if graceful shutdown cannot finish promptly."""
    grace = max(1.0, float(grace))

    def _watch() -> None:
        time.sleep(grace)
        if shutdown_event.is_set():
            print(
                f"[force shutdown] {label} still running after {grace:.0f}s; "
                "terminating process group.",
                flush=True,
            )
            terminate_active_subprocesses(grace=0.1)
            _force_exit_process_group(1)

    thread = threading.Thread(
        target=_watch,
        name=f"{label}-watchdog",
        daemon=True,
    )
    thread.start()
    return thread


def run_splits(
    split_names: List[str],
    dataset_name: str = "llm-srbench",
    cases: Optional[List[Dict[str, str]]] = None,
    equation_workers: int = 4,
    global_workers: int = 16,
    eval_owner_max_inflight: int = 32,
    eval_executor_log_interval: float = 10.0,
    dynamic_global_workers: bool = True,
    global_worker_initial: int = 1,
    global_worker_cpu_grow_threshold: float = 85.0,
    global_worker_cpu_shrink_threshold: float = 90.0,
    global_worker_cpu_check_interval: float = 0.1,
    **run_kwargs,
) -> Dict[str, Dict[str, dict]]:
    """
    Run pending equations using equation threads and one shared eval queue.

    Each equation still advances its own tree search sequentially, but all
    formula evaluations are submitted to one shared owner-aware global queue.
    """
    deprecated_global_workers = run_kwargs.pop("global_eval_workers", None)
    if deprecated_global_workers is not None and global_workers == 16:
        global_workers = int(deprecated_global_workers)

    dataset_name = normalize_dataset_name(dataset_name)
    equations_by_split: Dict[str, List[str]] = {}
    if cases is not None:
        for case in cases:
            if normalize_dataset_name(case["dataset"]) != dataset_name:
                raise ValueError("run_splits only supports one dataset family per invocation")
            split_name = case["split"]
            equations_by_split.setdefault(split_name, [])
            if case["equation"] not in equations_by_split[split_name]:
                equations_by_split[split_name].append(case["equation"])
        split_names = list(equations_by_split.keys())
        if not equations_by_split:
            return {}
        print(
            f"[Concurrent test] Selected {sum(len(v) for v in equations_by_split.values())} "
            f"case(s) from dataset '{dataset_name}'."
        )
    else:
        split_names = [s.strip() for s in split_names if s.strip()]
        if not split_names:
            return {}

        print(
            f"[Concurrent test] Pre-loading {len(split_names)} split(s) "
            f"from dataset '{dataset_name}' ..."
        )
        for split_name in split_names:
            print(f"  - {split_name}")
            equations = list_equations(dataset_name, split_name)
            if not equations:
                print(f"[Concurrent test] No equations found in split '{split_name}'.")
                continue
            equations_by_split[split_name] = equations

    if not equations_by_split:
        return {}

    log_base = Path(__file__).parent / "logs" / dataset_name
    _cfg_path = run_kwargs.get("llm_config_path")
    _model_tag = Path(_cfg_path).stem if _cfg_path else None
    _run_tag = _validate_log_component("run_tag", run_kwargs.get("run_tag"))
    completed_map = _find_completed_equations(
        log_base, model_tag=_model_tag, run_tag=_run_tag)
    if run_kwargs.get("resume"):
        ckpt_completed = _find_completed_checkpoints(
            _default_checkpoint_dir(
                dataset_name,
                _run_tag,
                _cfg_path,
                run_kwargs.get("checkpoint_dir"),
            )
        )
        completed_map.update({
            eq_tag: path
            for eq_tag, path in ckpt_completed.items()
            if eq_tag not in completed_map
        })

    pending: List[tuple[str, str]] = []
    skipped: List[tuple[str, str]] = []
    for split_name, equations in equations_by_split.items():
        for eq in equations:
            eq_tag = eq.replace("/", "_")
            if eq_tag in completed_map:
                skipped.append((split_name, eq))
            else:
                pending.append((split_name, eq))

    total = sum(len(v) for v in equations_by_split.values())
    print(
        f"[Concurrent test] Total {total} equations across {len(equations_by_split)} split(s), "
        f"{len(skipped)} completed, {len(pending)} pending, "
        f"equation_workers={equation_workers}, "
        f"global_workers={global_workers}, "
        f"eval_owner_max_inflight={eval_owner_max_inflight}, "
        f"dynamic_global_workers={dynamic_global_workers}"
    )
    for split_name, equations in equations_by_split.items():
        n_done = sum(
            1 for eq in equations
            if eq.replace("/", "_") in completed_map
        )
        print(f"\n=== {split_name}: {len(equations)} equations, {n_done} completed ===")
        for i, eq in enumerate(equations):
            eq_tag = eq.replace("/", "_")
            status = "✓ done" if eq_tag in completed_map else "  pending"
            print(f"  {i:>2d}. {eq}  {status}")
    print()

    if not pending:
        print("[Concurrent test] All equations completed, no re-run needed.")
        return {}

    results: Dict[str, Dict[str, dict]] = {
        split_name: {} for split_name in equations_by_split
    }
    completed_count = 0
    lock = threading.Lock()
    equation_state_lock = threading.Lock()
    equation_states: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()
    n_pending = len(pending)

    _preload_llm_sdk(run_kwargs.get("llm_config"))
    code_version = run_kwargs.get("code_version") or _get_code_version()
    run_call_kwargs = dict(run_kwargs, code_version=code_version)

    def _fmt_v(v: float) -> str:
        if v == float("inf") or v >= 1e9:
            return "—"
        return f"{v:.2e}"

    eval_executor = FairEvalExecutor(
        max_workers=global_workers,
        owner_max_inflight=eval_owner_max_inflight,
        dynamic_workers=dynamic_global_workers,
        initial_workers=global_worker_initial,
        cpu_grow_threshold=global_worker_cpu_grow_threshold,
        cpu_shrink_threshold=global_worker_cpu_shrink_threshold,
        cpu_check_interval=global_worker_cpu_check_interval,
    )
    shutdown_event = threading.Event()
    executor_log_path = _default_checkpoint_dir(
        dataset_name,
        _run_tag,
        _cfg_path,
        run_kwargs.get("checkpoint_dir"),
    ) / "fair_eval_executor.log"
    executor_log_stop, executor_log_thread = _start_eval_executor_logger(
        eval_executor,
        shutdown_event=shutdown_event,
        log_path=executor_log_path,
        interval=eval_executor_log_interval,
    )
    equation_log_path = executor_log_path.with_name("equation_scheduler.log")
    equation_log_stop, equation_log_thread = _start_equation_scheduler_logger(
        equation_states,
        equation_state_lock,
        shutdown_event=shutdown_event,
        log_path=equation_log_path,
        equation_workers=equation_workers,
        interval=eval_executor_log_interval,
    )
    equation_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, equation_workers))
    signal_count = 0
    signal_owner_pid = os.getpid()

    def _cleanup(signum=None, frame=None):
        nonlocal signal_count
        if os.getpid() != signal_owner_pid:
            if signum is not None:
                try:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
                except Exception:
                    pass
            os._exit(128 + int(signum or 0))
        signal_count += 1
        if signal_count == 1:
            print(
                f"\n[graceful shutdown] Received signal {signum}; "
                "saving checkpoints and stopping new equations. Send again to force exit.",
                flush=True,
            )
            shutdown_event.set()
            _start_shutdown_watchdog(
                shutdown_event,
                grace=300.0,
                label="run_splits graceful shutdown",
            )
            terminate_active_subprocesses(grace=0.5)
            eval_executor.shutdown(wait=False, cancel_futures=True)
            equation_pool.shutdown(wait=False, cancel_futures=True)
            return

        print("\n[force shutdown] Second signal received; cancelling workers.", flush=True)
        shutdown_event.set()
        terminate_active_subprocesses(grace=0.1)
        equation_pool.shutdown(wait=False, cancel_futures=True)
        eval_executor.shutdown(wait=False, cancel_futures=True)
        try:
            import psutil
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
        except Exception:
            pass
        _force_exit_process_group(1)
        os._exit(1)

    prev_sigint = signal.signal(signal.SIGINT, _cleanup)
    prev_sigterm = signal.signal(signal.SIGTERM, _cleanup)
    prev_sighup = signal.signal(signal.SIGHUP, _cleanup)

    def _equation_key(split_name: str, eq: str) -> str:
        return f"{split_name}:{eq}"

    def _set_equation_state(
        split_name: str,
        eq: str,
        state: str,
        *,
        error: Optional[str] = None,
    ) -> None:
        now_mono = time.monotonic()
        key = _equation_key(split_name, eq)
        with equation_state_lock:
            record = equation_states.setdefault(
                key,
                {
                    "split": split_name,
                    "equation": eq,
                    "submitted_at": now_mono,
                },
            )
            record["state"] = state
            record["updated_at"] = now_mono
            if state == "running":
                record.setdefault("started_at", now_mono)
            if state in {"done", "error"}:
                record["finished_at"] = now_mono
            if error:
                record["error"] = error[:200]

    def _run_equation_logged(split_name: str, eq: str) -> dict:
        _set_equation_state(split_name, eq, "running")
        try:
            result = run(
                dataset=dataset_name,
                split=split_name,
                equation=eq,
                eval_executor=eval_executor,
                eval_owner_id=f"{dataset_name}:{split_name}:{eq}",
                shutdown_event=shutdown_event,
                **run_call_kwargs,
            )
        except Exception as exc:
            _set_equation_state(split_name, eq, "error", error=str(exc))
            raise
        _set_equation_state(split_name, eq, "done")
        return result

    try:
        for split_name, eq in pending:
            _set_equation_state(split_name, eq, "submitted")
        futures = {
            equation_pool.submit(
                _run_equation_logged,
                split_name,
                eq,
            ): (split_name, eq)
            for split_name, eq in pending
        }
        for future in _iter_completed_futures(futures, shutdown_event):
            split_name, eq = futures[future]
            with lock:
                completed_count += 1
                idx = completed_count
            try:
                result = future.result()
                results[split_name][eq] = result
                test_v = result.get("test_nmse", float("inf"))
                ood_v = result.get("ood_test_nmse", float("inf"))
                expr = str(result.get("expression", "—"))[:60]
                print(
                    f"  [{idx}/{n_pending}] ✓ {eq} ({split_name}): "
                    f"test={_fmt_v(test_v)}  ood={_fmt_v(ood_v)}  → {expr}"
                )
            except Exception as e:
                results[split_name][eq] = {"error": str(e)}
                print(f"  [{idx}/{n_pending}] ✗ {eq} ({split_name}): {e}")
    finally:
        executor_log_stop.set()
        equation_log_stop.set()
        executor_log_thread.join(timeout=2.0)
        equation_log_thread.join(timeout=2.0)
        if shutdown_event.is_set():
            equation_pool.shutdown(wait=False, cancel_futures=True)
            eval_executor.shutdown(wait=False, cancel_futures=True)
        else:
            equation_pool.shutdown(wait=True)
            eval_executor.shutdown(wait=True)
        signal.signal(signal.SIGINT, prev_sigint)
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGHUP, prev_sighup)

    elapsed = time.time() - t_start

    print()
    print("=" * 100)
    print(
        f"Concurrent test results summary: {len(equations_by_split)} split(s), "
        f"{total} equations, {n_pending} run this time, {len(skipped)} skipped, "
        f"elapsed {elapsed:.1f}s"
    )
    print("=" * 100)
    print(f"{'split':<15s} {'equation':<35s} {'status':>8s} {'test_nmse':>12s} {'ood_nmse':>12s}  {'expression'}")
    print("-" * 100)
    for split_name, equations in equations_by_split.items():
        split_results = results.get(split_name, {})
        for eq in equations:
            eq_tag = eq.replace("/", "_")
            if eq_tag in completed_map:
                print(
                    f"{split_name:<15s} {eq:<35s} {'skip':>8s} "
                    f"{'—':>12s} {'—':>12s}  "
                    f"(existing result: {completed_map[eq_tag].name})"
                )
            elif eq in split_results:
                r = split_results[eq]
                if "error" in r:
                    print(
                        f"{split_name:<15s} {eq:<35s} {'fail':>8s} "
                        f"{'ERROR':>12s} {'':>12s}  {str(r['error'])[:40]}"
                    )
                else:
                    tv = r.get("test_nmse", float("inf"))
                    ov = r.get("ood_test_nmse", float("inf"))
                    expr = str(r.get("expression", "—"))[:50]
                    print(
                        f"{split_name:<15s} {eq:<35s} {'done':>8s} "
                        f"{_fmt_v(tv):>12s} {_fmt_v(ov):>12s}  {expr}"
                    )
    print("=" * 100)

    valid = [
        r
        for split_results in results.values()
        for r in split_results.values()
        if "error" not in r
    ]
    if valid:
        test_nmses = [
            r["test_nmse"]
            for r in valid
            if r.get("test_nmse", float("inf")) < float("inf")
        ]
        if test_nmses:
            print(f"  test_nmse  — median: {np.median(test_nmses):.2e}, mean: {np.mean(test_nmses):.2e}")
    print(f"  This run: success {len(valid)}/{n_pending}, failed {n_pending - len(valid)}/{n_pending}")
    print(f"  Total: completed {len(skipped) + len(valid)}/{total}")
    print()

    return results


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Symbolic regression framework based on LLM Selector+Generator evolution tree",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    default_global_workers = 400

    parser.add_argument(
        "--degenerated-generator", action="store_true",
        help="Degenerated Generator: use hardcoded single-variable seeds + empty description, no LLM calls",
    )
    parser.add_argument(
        "--degenerated-selector", action="store_true",
        help="Alias for --degenerated-selector1 (Boltzmann rank sampling, no LLM calls)",
    )
    parser.add_argument(
        "--degenerated-selector1", action="store_true",
        help="Degenerated Selector 1: Boltzmann rank sampling for parent selection, no LLM calls",
    )
    parser.add_argument(
        "--degenerated-selector2", action="store_true",
        help="Degenerated Selector 2: LLM selection; omit AST metrics and description from selector prompts",
    )
    parser.add_argument(
        "--degenerated-mutator1", action="store_true",
        help="Degenerated Mutator (option 1): use only programmatic mutations (delete/add), no LLM calls",
    )
    parser.add_argument(
        "--degenerated-mutator2", action="store_true",
        help="Degenerated Mutator (option 2): use only LLM mutations, no programmatic mutations",
    )
    parser.add_argument(
        "--degenerated-mutator3", action="store_true",
        help="Degenerated Mutator (option 3): same as option 2, plus strip AST blocks and AST-local guidance from LLM mutator prompts",
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Local path to the HF dataset repo (skip HuggingFace download).\n"
             "Should point to a directory containing data/*.parquet and lsr_bench_data.hdf5",
    )
    parser.add_argument(
        "--dataset", type=str, default=None, metavar="DATASET",
        help="Dataset family: llm-srbench | llm-srbench-noise1pct | llm-srbench-noise5pct | aifeynman.\n"
             "If --split/--equations are omitted, all splits in this dataset are evaluated.",
    )
    parser.add_argument(
        "--split", type=str, default=None,
        help="Comma-separated split list. llm-srbench: bio_pop_growth,chem_react,matsci,phys_osc,lsrtransform; "
             "aifeynman: feynmanequations,bonusequations.",
    )
    parser.add_argument(
        "--equations", "--equation", dest="equations", type=str, default=None,
        help="Comma-separated concrete equation names, e.g. 'BPG14,CRK2' or 'I.6.2,test_1'. "
             "When combined with --split, the final case set is their union.",
    )
    parser.add_argument("--equation-workers", type=int, default=4,
                        help="Max number of equations active at once (default: 4)")
    parser.add_argument("--global-workers", "--global-eval-workers",
                        dest="global_workers", type=int,
                        default=default_global_workers,
                        help="Hard cap for global candidate evaluation workers shared by all active equations "
                             f"(default: {default_global_workers}). "
                             "--global-eval-workers is kept as a deprecated alias.")
    parser.add_argument("--global-worker-initial", type=int, default=1,
                        help="Initial active global worker dispatch limit before CPU-guided ramp-up (default: 1)")
    parser.add_argument("--global-worker-cpu-grow-threshold", type=float, default=85.0,
                        help="Increase active global worker limit while smoothed CPU load is at or below this percent (default: 85)")
    parser.add_argument("--global-worker-cpu-shrink-threshold", type=float, default=90.0,
                        help="Decrease active global worker limit when smoothed CPU load reaches this percent (default: 90)")
    parser.add_argument("--global-worker-cpu-check-interval", type=float, default=0.1,
                        help="Seconds between CPU checks for dynamic global worker dispatch (default: 0.1)")
    parser.add_argument("--disable-dynamic-global-workers", action="store_true",
                        help="Disable CPU-guided dispatch throttling and use --global-workers as a fixed concurrency limit")
    parser.add_argument("--eval-owner-max-inflight", type=int, default=32,
                        help="Soft cap on concurrently running global eval tasks per equation owner; "
                             "0 disables the cap (default: 32)")
    parser.add_argument("--eval-executor-log-interval", type=float, default=10.0,
                        help="Seconds between FairEvalExecutor live counter logs (default: 10)")
    parser.add_argument("--list-equations", action="store_true", help="List all equation names in the dataset and exit")

    # LLM config
    parser.add_argument("--llm-config", type=str, default=None,
                        help="LLM config YAML file path (required when non-degenerated components exist).\n"
                             "Can specify model / base_url / temperature / total max_tokens\n"
                             "separately for generator / selector / mutator.\n"
                             "API Key is set via OPENAI_API_KEY environment variable")
    parser.add_argument("--verify-llm-config", type=str, default=None,
                        help="Optional LLM config YAML file path used only for GT equivalence verification.\n"
                             "If omitted, verification uses --llm-config.")
    parser.add_argument("--llm-max-retries", type=int, default=5,
                        help="Max LLM call attempts per request for generator/selector/mutator (default: 5)")

    # Search hyperparams
    parser.add_argument("--max-steps", type=int, default=30, help="Max evolution steps (default: 30)")
    parser.add_argument("--n-seeds", type=int, default=20, help="Number of initial seeds (default: 20)")
    parser.add_argument("--selector-context-size", type=int, default=1000,
                        help="Max number of nodes passed to Selector each time (default: 1000)")
    parser.add_argument("--candidate-num", type=int, default=5,
                        help="Number of parents selected by Selector per step (default: 5)")

    # Eval hyperparams
    parser.add_argument("--optimizer",
                        choices=["Structure", "DE", "CMA-ES", "L-BFGS-B", "least_squares"],
                        default="Structure", help="Constant optimizer (default: Structure)")
    parser.add_argument("--max-mature-nodes", type=int, default=5,
                        help="Early stop after collecting this many mature nodes (default: 5)")
    parser.add_argument("--mature-train-threshold", type=float, default=None,
                        help="Mature node train NMSE threshold (default: 1.05*GT train NMSE)")
    parser.add_argument("--mature-anneal-budget", type=int, default=0,
                        help="Times a mature node may still be selected as parent before exclusion (default: 0)")
    parser.add_argument("--overfit-min-depth", type=int, default=10,
                        help="Only check overfitting when SymPy AST depth >= this value (default: 10)")
    parser.add_argument("--mutator-seen-topk", type=int, default=100,
                        help="Number of historical best formulas visible to LLM Mutator (default: 100)")
    parser.add_argument("--max-params", type=int, default=10,
                        help="Max number of candidate formula params; skip evaluation if exceeded (default: 10)")
    parser.add_argument("--n-eval-workers", type=int, default=16,
                        help="Local child evaluation workers for single-equation runs without shared global workers (default: 16)")
    parser.add_argument("--timeout", dest="timeout",
                        type=float, default=120.0,
                        help="Per-task wall-clock timeout in seconds, shared by all "
                             "three pool task types (normalization, evaluation, "
                             "degeneracy check); also passed to the optimizer timeout "
                             "(default: 120).")

    parser.add_argument("--refine-output", action="store_true",
                        help="Enable final scan refinement (default off): ensure all final output nodes are selected as parent at least once and undergo mutation exploration")
    parser.add_argument("--enable-describe", action="store_true",
                        help="Enable LLM structural descriptions for nodes (default off): generate posterior natural-language descriptions fed to the selector")
    parser.add_argument("--preprocessing", action="store_true",
                        help="Legacy option for extracting preprocessing suggestions (default off); deterministic variable-transform expansion is disabled")
    parser.add_argument("--run-tag", type=str, default=None,
                        help="Experiment tag for isolating parallel runs.\n"
                             "Inserted into log path: logs/<dataset>/<run_tag>/<model>/<date>/.\n"
                             "run.sh defaults this to <mode>.\n"
                             "Also scopes checkpoint resume to the same tag.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume each equation from logs/<dataset>/<run_tag>/<model>/checkpoint/<equation>.json when available.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory (default: logs/<dataset>/<run_tag>/<model>/checkpoint).")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Echo verbose diagnostic output to console; EvalDiag is written under the checkpoint directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console verbosity; also disables --verbose diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Make main process the process group leader so Ctrl+C/kill terminates all child processes at once
    try:
        os.setpgrp()
    except OSError:
        pass

    args = _parse_args()

    if args.data_dir:
        set_repo_dir(args.data_dir)

    if args.list_equations:
        if not args.dataset:
            print("Please specify a dataset with --dataset, e.g.: --dataset llm-srbench")
            sys.exit(1)
        dataset_name = normalize_dataset_name(args.dataset)
        split_names = (
            [s.strip() for s in args.split.split(",") if s.strip()]
            if args.split else dataset_split_names(dataset_name)
        )
        if args.equations:
            equation_names = [e.strip() for e in args.equations.split(",") if e.strip()]
            try:
                cases = resolve_requested_cases(
                    dataset_name,
                    split_names=split_names if args.split else [],
                    equation_names=equation_names,
                )
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                sys.exit(1)
            for case in cases:
                print(f"  {case['equation']}  ({case['split']})")
            sys.exit(0)
        for split_name in split_names:
            print(f"\n=== {dataset_name}:{split_name} ===")
            for eq in list_equations(dataset_name, split_name):
                print(f"  {eq}")
        sys.exit(0)

    _llm_config = None
    if args.llm_config:
        _cfg_path = Path(args.llm_config)
        if _cfg_path.is_file():
            import yaml
            with open(_cfg_path, "r", encoding="utf-8") as f:
                _llm_config = yaml.safe_load(f) or {}
            print(f"[LLM Config] Loaded config: {_cfg_path}")
        else:
            print(f"Error: LLM config file '{_cfg_path}' does not exist.")
            sys.exit(1)

    _verify_llm_config = None
    if args.verify_llm_config:
        _verify_cfg_path = Path(args.verify_llm_config)
        if _verify_cfg_path.is_file():
            import yaml
            with open(_verify_cfg_path, "r", encoding="utf-8") as f:
                _verify_llm_config = yaml.safe_load(f) or {}
            print(f"[Verify LLM Config] Loaded config: {_verify_cfg_path}")
        else:
            print(f"Error: verify LLM config file '{_verify_cfg_path}' does not exist.")
            sys.exit(1)

    run_verbose = bool(args.verbose and not args.quiet)

    common_kwargs = dict(
        llm_config=_llm_config,
        llm_config_path=args.llm_config if _llm_config else None,
        llm_max_retries=args.llm_max_retries,
        verify_llm_config=_verify_llm_config,
        verify_llm_config_path=args.verify_llm_config if _verify_llm_config else None,
        degenerated_generator=args.degenerated_generator,
        degenerated_selector1=(
            args.degenerated_selector1 or args.degenerated_selector
        ),
        degenerated_selector2=args.degenerated_selector2,
        degenerated_mutator1=args.degenerated_mutator1,
        degenerated_mutator2=args.degenerated_mutator2,
        degenerated_mutator3=args.degenerated_mutator3,
        max_steps=args.max_steps,
        n_seeds=args.n_seeds,
        selector_context_size=args.selector_context_size,
        candidate_num=args.candidate_num,
        optimizer=args.optimizer,
        max_mature_nodes=args.max_mature_nodes,
        mature_train_threshold=args.mature_train_threshold,
        mature_anneal_budget=args.mature_anneal_budget,
        overfit_min_depth=args.overfit_min_depth,
        n_eval_workers=args.n_eval_workers,
        timeout=args.timeout,
        mutator_seen_topk=args.mutator_seen_topk,
        max_params=args.max_params,
        refine_output=args.refine_output,
        enable_describe=args.enable_describe,
        enable_preprocessing=args.preprocessing,
        run_tag=args.run_tag,
        resume=args.resume,
        checkpoint_dir=args.checkpoint_dir,
        verbose=run_verbose,
    )
    common_kwargs["code_version"] = _get_code_version()

    _preload_llm_sdk(_llm_config)

    main_shutdown_event = threading.Event()
    main_signal_owner_pid = os.getpid()

    def _main_graceful_signal(signum=None, frame=None):
        if os.getpid() != main_signal_owner_pid:
            if signum is not None:
                try:
                    signal.signal(signum, signal.SIG_DFL)
                    os.kill(os.getpid(), signum)
                except Exception:
                    pass
            os._exit(128 + int(signum or 0))
        nonlocal_main = getattr(_main_graceful_signal, "_count", 0) + 1
        setattr(_main_graceful_signal, "_count", nonlocal_main)
        if nonlocal_main == 1:
            print(
                f"\n[graceful shutdown] Received signal {signum}; "
                "current equation will checkpoint before stopping. Send again to force exit.",
                flush=True,
            )
            main_shutdown_event.set()
            _start_shutdown_watchdog(
                main_shutdown_event,
                grace=300.0,
                label="main graceful shutdown",
            )
            terminate_active_subprocesses(grace=0.5)
            return
        print("\n[force shutdown] Second signal received; terminating process group.", flush=True)
        terminate_active_subprocesses(grace=0.1)
        _force_exit_process_group(1)

    signal.signal(signal.SIGINT, _main_graceful_signal)
    signal.signal(signal.SIGTERM, _main_graceful_signal)

    if args.dataset or args.split or args.equations:
        dataset_name = normalize_dataset_name(args.dataset or "llm-srbench")
        split_names = (
            [s.strip() for s in args.split.split(",") if s.strip()]
            if args.split else []
        )
        equation_names = (
            [e.strip() for e in args.equations.split(",") if e.strip()]
            if args.equations else []
        )
        try:
            cases = resolve_requested_cases(
                dataset_name,
                split_names=split_names,
                equation_names=equation_names,
            )
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)

        if not cases:
            print("No valid equations found.")
            sys.exit(1)

        print(
            f"\n[Eval] Dataset={dataset_name}, total {len(cases)} case(s), "
            f"equation_workers={args.equation_workers}, "
            f"global_workers={args.global_workers}, "
            f"eval_owner_max_inflight={args.eval_owner_max_inflight}, "
            f"dynamic_global_workers={not args.disable_dynamic_global_workers}"
        )
        for i, case in enumerate(cases):
            print(f"  {i:>3d}. {case['equation']}  ({case['split']})")
        print()

        if len(cases) == 1:
            case = cases[0]
            run(
                dataset=case["dataset"],
                split=case["split"],
                equation=case["equation"],
                shutdown_event=main_shutdown_event,
                **common_kwargs,
            )
        else:
            run_splits(
                split_names=[],
                dataset_name=dataset_name,
                cases=cases,
                equation_workers=args.equation_workers,
                global_workers=args.global_workers,
                eval_owner_max_inflight=args.eval_owner_max_inflight,
                eval_executor_log_interval=args.eval_executor_log_interval,
                dynamic_global_workers=not args.disable_dynamic_global_workers,
                global_worker_initial=args.global_worker_initial,
                global_worker_cpu_grow_threshold=args.global_worker_cpu_grow_threshold,
                global_worker_cpu_shrink_threshold=args.global_worker_cpu_shrink_threshold,
                global_worker_cpu_check_interval=args.global_worker_cpu_check_interval,
                **common_kwargs,
            )
    else:
        # No --dataset and no --equation -> use built-in demo data
        run(
            dataset=None,
            equation=None,
            shutdown_event=main_shutdown_event,
            **common_kwargs,
        )
