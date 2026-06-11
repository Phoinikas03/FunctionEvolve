"""
Evolution tree search scheduler.

Search Strategy
---------------
Maintains an evolution tree (EvolutionTree), where each node stores:
  - symbolic formula (skeleton_str)
  - AST structural features (ast_features)
  - natural language description of the structure (description)
  - evaluation score NMSE

Each iteration proceeds as follows:
  1. Selector LLM observes the full tree summary and selects a parent node
  2. ASTMutator performs programmatic mutations on the parent (subtree deletion + template term addition)
  3. LLMMutator suggests 20 additional non-template structural improvements
  4. All candidate children are evaluated with constant optimization, then attached under the parent
"""

from __future__ import annotations

import concurrent.futures
import multiprocessing
import os
import re
import signal
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, TextIO, Tuple

import sympy as sp

from .checkpoint import update_manifest, utc_timestamp, write_checkpoint_atomic
from .dataset import SRDataset
from .degeneration import DegenerationConfig, DegenerationEngine
from .evaluator import Evaluator, EvalResult
from .evolution_tree import EvolutionTree, EvolutionNode
from .mutator import ASTMutator
from .normalization import ExpressionNormalizer

_ACTIVE_SUBPROCESSES: Set[multiprocessing.Process] = set()
_ACTIVE_SUBPROCESSES_LOCK = threading.Lock()
_ACTIVE_PHASES: Counter[str] = Counter()
_ACTIVE_PHASES_LOCK = threading.Lock()
_TERMINAL_CANDIDATE_STATUSES = {
    "evaluated",
    "failed",
    "exception",
    "interrupted",
    "skipped",
}
_ACTIVE_CANDIDATE_STATUSES = {
    "pending",
    "queued",
    "running",
}


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


def _reset_child_signal_handlers() -> None:
    """Let forked helper processes terminate silently on parent cleanup."""
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(signum, signal.SIG_DFL)
        except (OSError, ValueError):
            pass


def _register_subprocess(proc: multiprocessing.Process) -> None:
    with _ACTIVE_SUBPROCESSES_LOCK:
        _ACTIVE_SUBPROCESSES.add(proc)


def _unregister_subprocess(proc: multiprocessing.Process) -> None:
    with _ACTIVE_SUBPROCESSES_LOCK:
        _ACTIVE_SUBPROCESSES.discard(proc)


def _active_subprocess_count() -> int:
    with _ACTIVE_SUBPROCESSES_LOCK:
        return len(_ACTIVE_SUBPROCESSES)


def _failed_eval_result(reason: str) -> EvalResult:
    return EvalResult(
        None,
        [],
        float("inf"),
        float("inf"),
        float("inf"),
        reason,
    )


def _enter_phase(name: str) -> None:
    with _ACTIVE_PHASES_LOCK:
        _ACTIVE_PHASES[name] += 1


def _exit_phase(name: str) -> None:
    with _ACTIVE_PHASES_LOCK:
        _ACTIVE_PHASES[name] -= 1
        if _ACTIVE_PHASES[name] <= 0:
            _ACTIVE_PHASES.pop(name, None)


def _phase_snapshot() -> Dict[str, int]:
    with _ACTIVE_PHASES_LOCK:
        return dict(_ACTIVE_PHASES)


def active_phase_snapshot() -> Dict[str, int]:
    """Return live counts for fork/child/inline evaluation phases."""
    return _phase_snapshot()


def _stop_process(proc: multiprocessing.Process, grace: float = 0.5) -> None:
    """Terminate one evaluation subprocess, escalating to kill if needed."""
    try:
        if not proc.is_alive():
            return
    except Exception:
        return

    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.join(timeout=grace)
    except Exception:
        pass

    try:
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=grace)
    except Exception:
        pass


def terminate_active_subprocesses(grace: float = 0.5) -> None:
    """Best-effort cleanup for forked evaluator/degeneracy subprocesses."""
    with _ACTIVE_SUBPROCESSES_LOCK:
        procs = list(_ACTIVE_SUBPROCESSES)
    for proc in procs:
        _stop_process(proc, grace=grace)


def _join_process_interruptibly(
    proc: multiprocessing.Process,
    timeout: float,
    shutdown_event: Optional[threading.Event] = None,
) -> bool:
    """Join a subprocess until it exits or reaches its hard timeout.

    Graceful shutdown stops already-started helpers too; otherwise executor
    threads can sit in joins long after the main process has checkpointed.
    """
    deadline = time.monotonic() + timeout
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            _stop_process(proc)
            return False
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _stop_process(proc)
            return False
        proc.join(timeout=min(0.2, remaining))
        if not proc.is_alive():
            return True


def _prepare_candidate_payload(
    feature_names: List[str],
    expr_str: str,
    param_names: list,
    *,
    compact: bool = False,
) -> Tuple[str, Optional[sp.Expr], List[str], str]:
    """Normalize one candidate and compute its structural key off the equation thread."""
    normalizer = ExpressionNormalizer(feature_names)
    return _prepare_candidate_with_normalizer(
        normalizer, expr_str, param_names, compact=compact)


def _prepare_candidate_with_normalizer(
    normalizer: ExpressionNormalizer,
    expr_str: str,
    param_names: list,
    *,
    compact: bool = False,
    deadline: Optional[float] = None,
) -> Tuple[str, Optional[sp.Expr], List[str], str]:
    """Normalize one candidate with a caller-owned normalizer."""
    formula = str(expr_str or "").strip()
    params = list(param_names or [])
    if not formula:
        return "", None, params, ""
    try:
        formula, params = normalizer.normalize_expression(
            formula, compact=compact, deadline=deadline)
    except Exception:
        pass
    sp_expr = normalizer.parse(formula)
    if sp_expr is None:
        return "", None, params, ""
    try:
        key = normalizer.structural_key(formula)
    except Exception:
        key = formula
    return formula, sp_expr, params, key or formula


class _NormalizeSoftDeadline(BaseException):
    """Raised in the normalize child when its own soft deadline elapses.

    Subclasses ``BaseException`` (not ``Exception``) so the broad ``except
    Exception`` guards inside ``_prepare_candidate_with_normalizer`` cannot
    swallow it -- it must propagate up to the soft-landing handler.
    """


def _normalize_candidate_target(
    conn,
    feature_names,
    expr_str,
    param_names,
    compact,
    budget: float = 30.0,
    key_headroom: float = 5.0,
):
    """Normalize one candidate in an isolated subprocess, landing gracefully.

    The subprocess owns its own soft deadline instead of relying solely on the
    parent's hard kill:

      * Phase 1 spends ``budget - key_headroom`` seconds on the (deadline-aware)
        normalization.  The cooperative deadline inside ``select_normal_form``
        usually returns a best-effort normal form before the SIGALRM fires; the
        alarm is only the backstop for a single rewrite that runs long.
      * Phase 2 (the A path) spends the remaining ``key_headroom`` seconds
        computing just the cheap structural key on the *raw* expression, so a
        timed-out candidate still gets a real dedup fingerprint
        (``timeout_softkey``) instead of becoming a ``raw:`` orphan.
      * Only if even the key computation overruns do we fall back to a ``raw:``
        key (``timeout_raw``), matching the old behavior.

    The parent keeps a slightly longer hard-kill timeout as the ultimate
    backstop for a child stuck in non-interruptible C code.
    """
    _reset_child_signal_handlers()
    params = list(param_names or [])
    raw_formula = str(expr_str or "").strip()
    normalizer = ExpressionNormalizer(feature_names)
    sp_expr_raw = normalizer.parse(raw_formula) if raw_formula else None

    def _raise_soft_deadline(signum, frame):
        raise _NormalizeSoftDeadline()

    try:
        signal.signal(signal.SIGALRM, _raise_soft_deadline)
    except (OSError, ValueError):
        # Signals unavailable here: run once without the soft-landing budget.
        try:
            formula, sp_expr, params, key = _prepare_candidate_with_normalizer(
                normalizer, expr_str, param_names, compact=compact)
            conn.send((formula, sp_expr, params, key, "ok"))
        except Exception as exc:
            conn.send(("", None, params, "", f"exception:{exc}"))
        finally:
            conn.close()
        return

    norm_budget = max(1.0, float(budget) - float(key_headroom))
    deadline = time.monotonic() + norm_budget

    # Phase 1: deadline-aware full normalization.
    try:
        signal.setitimer(signal.ITIMER_REAL, norm_budget)
        formula, sp_expr, params, key = _prepare_candidate_with_normalizer(
            normalizer, expr_str, param_names, compact=compact, deadline=deadline)
        signal.setitimer(signal.ITIMER_REAL, 0)
        conn.send((formula, sp_expr, params, key, "ok"))
        conn.close()
        return
    except _NormalizeSoftDeadline:
        pass
    except Exception as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        conn.send(("", None, params, "", f"exception:{exc}"))
        conn.close()
        return

    # Phase 2 (A path): soft deadline hit -> spend the headroom on just the
    # structural key, computed from the *raw* expression so the fingerprint is
    # deterministic regardless of how far normalization had progressed.
    try:
        signal.setitimer(signal.ITIMER_REAL, max(1.0, float(key_headroom)))
        key = normalizer.structural_key(raw_formula)
        signal.setitimer(signal.ITIMER_REAL, 0)
        conn.send((raw_formula, sp_expr_raw, params, key, "timeout_softkey"))
    except _NormalizeSoftDeadline:
        conn.send((raw_formula, sp_expr_raw, params, f"raw:{raw_formula}", "timeout_raw"))
    except Exception as exc:
        conn.send(
            (raw_formula, sp_expr_raw, params, f"raw:{raw_formula}",
             f"timeout_raw:{exc}")
        )
    finally:
        try:
            signal.setitimer(signal.ITIMER_REAL, 0)
        except Exception:
            pass
        conn.close()


def _raw_candidate_target(conn, feature_names, expr_str, param_names):
    """Parse an unnormalized candidate in an isolated subprocess."""
    _reset_child_signal_handlers()
    formula = str(expr_str or "").strip()
    params = list(param_names or [])
    try:
        if not formula:
            conn.send(("", None, params, "", "empty"))
            return
        normalizer = ExpressionNormalizer(feature_names)
        sp_expr = normalizer.parse(formula)
        if sp_expr is None:
            conn.send(("", None, params, "", "parse_failed"))
            return
        # Avoid structural_key here: the normalization path has already timed
        # out, and structural_key also performs symbolic reparameterization.
        conn.send((formula, sp_expr, params, f"raw:{formula}", "timeout_raw"))
    except Exception as exc:
        conn.send(("", None, params, "", f"raw_exception:{exc}"))
    finally:
        conn.close()


def _raw_candidate_fallback(
    feature_names: List[str],
    expr_str: str,
    param_names: List[str],
    timeout: float,
    shutdown_event: Optional[threading.Event] = None,
) -> Tuple[str, Optional[sp.Expr], List[str], str, str]:
    """Return the raw expression after normalization timeout, still timeout-bound."""
    if shutdown_event is not None and shutdown_event.is_set():
        return ("", None, list(param_names or []), "", "interrupted")

    ctx = multiprocessing.get_context("fork")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_raw_candidate_target,
        args=(send_conn, feature_names, expr_str, param_names),
    )
    try:
        _enter_phase("raw_parse_starting")
        proc.start()
    finally:
        _exit_phase("raw_parse_starting")
    _register_subprocess(proc)
    send_conn.close()

    try:
        _enter_phase("raw_parse_child")
        completed = _join_process_interruptibly(
            proc,
            timeout=max(1.0, timeout),
            shutdown_event=shutdown_event,
        )
    finally:
        _exit_phase("raw_parse_child")
        _unregister_subprocess(proc)

    if not completed:
        recv_conn.close()
        return ("", None, list(param_names or []), "", "raw_timeout")

    try:
        result = recv_conn.recv() if recv_conn.poll() else None
        if not isinstance(result, tuple) or len(result) != 5:
            return ("", None, list(param_names or []), "", "raw_no_result")
        return result
    except Exception as exc:
        return ("", None, list(param_names or []), "", f"raw_exception:{exc}")
    finally:
        recv_conn.close()


def _normalize_candidate(
    feature_names: List[str],
    expr_str: str,
    param_names: List[str],
    compact: bool = False,
    normalize_timeout: float = 30.0,
    shutdown_event: Optional[threading.Event] = None,
) -> Tuple[str, Optional[sp.Expr], List[str], str, str]:
    if shutdown_event is not None and shutdown_event.is_set():
        return ("", None, list(param_names or []), "", "interrupted")

    # The child owns a soft deadline: it normalizes for (budget - key_headroom)
    # then spends key_headroom landing on just the structural key.  The parent's
    # hard-kill timeout is the child budget plus slack, so it only triggers when
    # the child is stuck in non-interruptible C code (rare); then we still fall
    # back to the raw-parse path below.
    key_headroom = min(5.0, max(1.0, normalize_timeout / 3.0))
    ctx = multiprocessing.get_context("fork")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_normalize_candidate_target,
        args=(send_conn, feature_names, expr_str, param_names, compact,
              float(max(1.0, normalize_timeout)), float(key_headroom)),
    )

    try:
        _enter_phase("normalize_starting")
        proc.start()
    finally:
        _exit_phase("normalize_starting")

    _register_subprocess(proc)
    send_conn.close()

    try:
        _enter_phase("normalize_child")
        completed = _join_process_interruptibly(
            proc,
            timeout=max(1.0, normalize_timeout) + 5.0,
            shutdown_event=shutdown_event,
        )
    finally:
        _exit_phase("normalize_child")
        _unregister_subprocess(proc)

    if not completed:
        recv_conn.close()
        return _raw_candidate_fallback(
            feature_names,
            expr_str,
            param_names,
            timeout=min(5.0, max(1.0, normalize_timeout)),
            shutdown_event=shutdown_event,
        )

    try:
        result = recv_conn.recv() if recv_conn.poll() else None
        if not isinstance(result, tuple) or len(result) != 5:
            return ("", None, list(param_names or []), "", "no_result")
        return result
    except Exception as exc:
        return ("", None, list(param_names or []), "", f"exception:{exc}")
    finally:
        recv_conn.close()


EvalCandidateResult = Tuple[str, List[str], str, EvalResult, List[float]]
# (status, simp_expr_str, simp_params, simp_key, reasons, simp_sp_expr)
DegenResult = Tuple[str, Optional[str], Optional[List[str]], str, List[str], Optional[sp.Expr]]


def _failed_eval_candidate_result(
    expr_str: str,
    param_names: list,
    reason: str,
) -> EvalCandidateResult:
    raw_expr = str(expr_str or "").strip()
    raw_param_names = list(param_names or [])
    return (
        raw_expr,
        raw_param_names,
        raw_expr,
        _failed_eval_result(reason),
        [],
    )


def _eval_candidate_payload(
    evaluator: Evaluator,
    sp_expr: sp.Expr,
    norm_expr: str,
    param_names: list,
    parent_params=None,
    shutdown_event: Optional[threading.Event] = None,
) -> EvalCandidateResult:
    """Evaluate one pre-normalized candidate. sp_expr and norm_expr come from the prepare stage."""
    normalizer = ExpressionNormalizer(evaluator.feature_names)
    norm_param_names = list(param_names)
    try:
        expr_key = normalizer.structural_key(norm_expr)
    except Exception:
        expr_key = norm_expr

    if shutdown_event is not None and shutdown_event.is_set():
        return norm_expr, norm_param_names, expr_key, _failed_eval_result("interrupted"), []

    try:
        result = evaluator.evaluate_skeleton(sp_expr, norm_param_names, parent_params)
    except Exception:
        result = _failed_eval_result("evaluator_exception")
    if result.train_nmse == float("inf"):
        return norm_expr, norm_param_names, expr_key, result, []
    if shutdown_event is not None and shutdown_event.is_set():
        return norm_expr, norm_param_names, expr_key, _failed_eval_result("interrupted"), []

    _enter_phase("normalize_fitted_params")
    try:
        norm_params = normalizer.normalize_fitted_params(
            norm_expr, norm_param_names, list(result.best_params))
    finally:
        _exit_phase("normalize_fitted_params")
    return norm_expr, norm_param_names, expr_key, result, norm_params


def _eval_candidate_target(conn, *args):
    """Run candidate optimization in one isolated subprocess."""
    _reset_child_signal_handlers()
    try:
        conn.send(_eval_candidate_payload(*args))
    except Exception:
        norm_expr = str(args[2]) if len(args) > 2 else ""
        param_names = list(args[3] or []) if len(args) > 3 else []
        conn.send(_failed_eval_candidate_result(
            norm_expr, param_names, "child_exception"))
    finally:
        conn.close()


def _eval_candidate_single(
    evaluator: Evaluator,
    sp_expr: sp.Expr,
    norm_expr: str,
    param_names: list,
    parent_params=None,
    timeout: float = 120,
    shutdown_event: Optional[threading.Event] = None,
) -> EvalCandidateResult:
    """Evaluate one pre-normalized candidate using a hard-timeout subprocess."""
    if shutdown_event is not None and shutdown_event.is_set():
        return _failed_eval_candidate_result(norm_expr, param_names, "interrupted")

    ctx = multiprocessing.get_context("fork")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_eval_candidate_target,
        args=(send_conn, evaluator, sp_expr, norm_expr, param_names, parent_params, shutdown_event),
    )
    try:
        _enter_phase("eval_starting")
        proc.start()
    finally:
        _exit_phase("eval_starting")
    _register_subprocess(proc)
    send_conn.close()

    try:
        _enter_phase("eval_child")
        completed = _join_process_interruptibly(
            proc, timeout=timeout, shutdown_event=shutdown_event)
    finally:
        _exit_phase("eval_child")
        _unregister_subprocess(proc)

    if not completed:
        recv_conn.close()
        reason = (
            "interrupted"
            if shutdown_event is not None and shutdown_event.is_set()
            else "child_timeout"
        )
        return _failed_eval_candidate_result(norm_expr, param_names, reason)

    try:
        result = recv_conn.recv() if recv_conn.poll() else (
            _failed_eval_candidate_result(
                norm_expr, param_names, "child_no_result")
        )
    except (EOFError, OSError):
        result = _failed_eval_candidate_result(
            norm_expr, param_names, "child_recv_error")
    finally:
        recv_conn.close()
    return result


def _degen_candidate_single(
    evaluator: Evaluator,
    mutator: ASTMutator,
    expr_str: str,
    param_names: list,
    fitted_params: List[float],
    timeout: float = 120,
    shutdown_event: Optional[threading.Event] = None,
) -> DegenResult:
    """Run degeneration checking after optimization has already completed."""
    if shutdown_event is not None and shutdown_event.is_set():
        return "interrupted", None, None, "", [], None

    normalizer = getattr(
        mutator, "normalizer", ExpressionNormalizer(evaluator.feature_names))
    degeneration_engine = DegenerationEngine(
        evaluator.feature_names,
        X_train=evaluator.X_train,
        config=DegenerationConfig(
            overfit_min_depth=getattr(mutator, "OVERFIT_MIN_DEPTH", 10),
        ),
        normalizer=normalizer,
    )
    (
        degen_status,
        simp_expr,
        simp_params,
        simp_key,
        degen_reasons,
        simp_sp_expr,
    ) = _check_degeneracy_safe(
        degeneration_engine,
        expr_str,
        param_names,
        fitted_params,
        timeout=timeout,
        shutdown_event=shutdown_event,
    )
    return degen_status, simp_expr, simp_params, simp_key, list(degen_reasons or []), simp_sp_expr


def _degeneracy_target(conn, degeneration_engine, expr_str, param_names, fitted_params):
    """Run check_degeneracy in a forked subprocess."""
    _reset_child_signal_handlers()
    try:
        report = degeneration_engine.analyze(expr_str, param_names, fitted_params)
        if report.status == "simplified" and report.first_child is not None:
            child = report.first_child
            result = (
                report.status,
                child.expression,
                child.params,
                report.canonical_key or "",
                report.reasons,
                report.canonical_sp_expr,
            )
        else:
            result = (
                report.status,
                None,
                None,
                report.canonical_key or "",
                report.reasons,
                report.canonical_sp_expr,
            )
        conn.send(result)
    except Exception:
        conn.send(("ok", None, None, "", [], None))
    finally:
        conn.close()


def _check_degeneracy_safe(degeneration_engine, expr_str, param_names, fitted_params,
                           timeout=30,
                           shutdown_event: Optional[threading.Event] = None):
    """check_degeneracy with a hard process-level timeout.

    SymPy's ``simplify()`` / ``equals()`` can hang indefinitely on certain
    expressions.  This wrapper runs the check in an isolated subprocess and
    kills it if the wall-clock *timeout* is exceeded.
    """
    _OK = ("ok", None, None, "", [], None)
    if shutdown_event is not None and shutdown_event.is_set():
        return ("interrupted", None, None, "", [], None)

    ctx = multiprocessing.get_context("fork")
    recv_conn, send_conn = ctx.Pipe(duplex=False)

    proc = ctx.Process(
        target=_degeneracy_target,
        args=(send_conn, degeneration_engine, expr_str, param_names, fitted_params),
    )
    proc.start()
    _register_subprocess(proc)
    send_conn.close()

    try:
        _enter_phase("degen_child")
        completed = _join_process_interruptibly(
            proc, timeout=timeout, shutdown_event=shutdown_event)
    finally:
        _exit_phase("degen_child")
        _unregister_subprocess(proc)

    if not completed:
        recv_conn.close()
        return ("timeout", None, None, "", [], None)

    try:
        result = recv_conn.recv() if recv_conn.poll() else _OK
    except (EOFError, OSError):
        result = _OK
    finally:
        recv_conn.close()

    return result


def _default_pool_workers() -> int:
    """Default candidate evaluation worker count: min(cpu_count, 8)."""
    return max(1, min(os.cpu_count() or 4, 8))


class TreeSearch:
    """Evolution tree search based on Selector + ASTMutator + LLMMutator + Generator."""

    MATURE_TRAIN_THRESHOLD_DEFAULT = 1e-11
    MATURE_TEST_THRESHOLD_DEFAULT = 1e-11

    def __init__(
        self,
        dataset: SRDataset,
        evaluator: Evaluator,
        tree: EvolutionTree,
        selector: Any,
        generator: Any,
        llm_mutator: Any,
        skip_programmatic_mutations: bool = False,
        max_steps: int = 30,
        n_seeds: int = 20,
        selector_context_size: int = 20,
        candidate_num: int = 5,
        max_mature_nodes: int = 5,
        mature_train_threshold: Optional[float] = None,
        mature_test_threshold: Optional[float] = None,
        mature_anneal_budget: int = 3,
        overfit_min_depth: Optional[int] = None,
        n_parent_workers: int = 4,
        n_eval_workers: Optional[int] = None,
        timeout: float = 120.0,
        degeneracy_timeout: Optional[float] = None,
        mutator_seen_topk: int = 100,
        max_params: int = 10,
        refine_output: bool = False,
        enable_describe: bool = False,
        verbose: bool = False,
        log_path: Optional[str] = None,
        eval_executor: Optional[Any] = None,
        eval_owner_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        diagnostic_log_path: Optional[str] = None,
        checkpoint_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_interval: float = 600.0,
        shutdown_event: Optional[threading.Event] = None,
    ):
        self.dataset = dataset
        self.evaluator = evaluator
        self.tree = tree
        self.selector = selector
        self.generator = generator
        self.llm_mutator = llm_mutator
        self.skip_programmatic_mutations = skip_programmatic_mutations
        self.max_steps = max_steps
        self.n_seeds = n_seeds
        self.selector_context_size = selector_context_size
        self.candidate_num = candidate_num
        self.max_mature_nodes = max_mature_nodes
        self.mature_train_threshold = (
            mature_train_threshold if mature_train_threshold is not None
            else self.MATURE_TRAIN_THRESHOLD_DEFAULT
        )
        # Kept for CLI/API/log compatibility. Test NMSE is not used by search logic.
        self.mature_test_threshold = (
            mature_test_threshold if mature_test_threshold is not None
            else self.MATURE_TEST_THRESHOLD_DEFAULT
        )
        self.verbose = verbose
        self.n_parent_workers = max(1, n_parent_workers)
        self.n_eval_workers = (
            n_eval_workers if n_eval_workers is not None
            else _default_pool_workers()
        )
        self.timeout = timeout
        # A single timeout drives all three shared-pool task types
        # (normalization / evaluation / degeneracy check).  degeneracy_timeout
        # may still be overridden explicitly (e.g. in tests); otherwise it
        # tracks timeout, and normalization uses the same budget.
        self.degeneracy_timeout = (
            degeneracy_timeout if degeneracy_timeout is not None else timeout
        )
        self.normalize_timeout = timeout
        self.mutator_seen_topk = mutator_seen_topk
        self.max_params = max_params
        self.refine_output = refine_output
        self.enable_describe = enable_describe
        self.eval_executor = eval_executor
        self.eval_owner_id = (
            eval_owner_id
            or dataset.equation_name
            or "default"
        )
        self.checkpoint_path = checkpoint_path
        self.checkpoint_metadata = checkpoint_metadata or {}
        self.checkpoint_interval = max(0.0, float(checkpoint_interval))
        self._last_checkpoint_monotonic = 0.0
        self.shutdown_event = shutdown_event

        self.normalizer = ExpressionNormalizer(dataset.feature_names)
        self.mutator = ASTMutator(dataset.feature_names, overfit_min_depth=overfit_min_depth)
        if hasattr(self.eval_executor, "register_owner_context"):
            self.eval_executor.register_owner_context(
                self.eval_owner_id,
                evaluator=self.evaluator,
                mutator=self.mutator,
                feature_names=list(dataset.feature_names),
            )
        self._lock = threading.RLock()

        self._log_file: Optional[TextIO] = None
        if log_path:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "w", encoding="utf-8")
        self._diagnostic_log_file: Optional[TextIO] = None
        if diagnostic_log_path:
            Path(diagnostic_log_path).parent.mkdir(parents=True, exist_ok=True)
            self._diagnostic_log_file = open(
                diagnostic_log_path, "a", encoding="utf-8"
            )
            self._diagnostic_log_file.write(
                f"\n[diagnostics] started_at={utc_timestamp()} owner={self.eval_owner_id}\n"
            )
            self._diagnostic_log_file.flush()

        self.param_map: Dict[str, List[str]] = {}
        self._selection_history: List[List[str]] = []
        self._seen_keys: Set[str] = set()
        self._preprocessing_rules: List[Dict[str, str]] = []
        self.history: List[Dict[str, Any]] = []
        self._mature_select_count: Dict[str, int] = {}
        self._current_mature_formulas: Set[str] = set()
        self._reported_mature_formulas: Set[str] = set()
        self.mature_anneal_budget = max(0, int(mature_anneal_budget))
        self._seed_initialized = False
        self._resume_step = 0
        self._checkpoint_stage = "created"
        self._candidate_statuses: Dict[str, str] = {}
        self._active_step: Optional[Dict[str, Any]] = None
        self._final_sweep_state: Optional[Dict[str, Any]] = None
        self._last_eval_diagnostic_monotonic = 0.0

    def _submit_candidate_eval(self, executor: Any, *args: Any):
        """Submit candidate optimization to either a local or shared executor."""
        if self.shutdown_requested():
            raise RuntimeError("shutdown requested; refusing eval submission")
        if hasattr(executor, "submit_eval"):
            return executor.submit_eval(
                self.eval_owner_id,
                _eval_candidate_single,
                *args,
                task_priority=0,
                shutdown_event=self.shutdown_event,
            )
        return executor.submit(
            _eval_candidate_single,
            *args,
            shutdown_event=self.shutdown_event,
        )

    def _submit_degeneracy_check(self, executor: Any, *args: Any):
        """Submit post-eval degeneration checking as a separate task."""
        if self.shutdown_requested():
            raise RuntimeError("shutdown requested; refusing degen submission")
        if hasattr(executor, "submit_eval"):
            return executor.submit_eval(
                self.eval_owner_id,
                _degen_candidate_single,
                *args,
                task_priority=1,
                shutdown_event=self.shutdown_event,
            )
        return executor.submit(
            _degen_candidate_single,
            *args,
            shutdown_event=self.shutdown_event,
        )

    def _submit_normalization(self, executor: Any, *args: Any):
        """Submit one candidate for normalization in the eval process pool."""
        if self.shutdown_requested():
            raise RuntimeError("shutdown requested; refusing normalization")
        if hasattr(executor, "submit_eval"):
            return executor.submit_eval(
                self.eval_owner_id,
                _normalize_candidate,
                *args,
                task_priority=1,
                shutdown_event=self.shutdown_event,
            )
        return executor.submit(
            _normalize_candidate,
            *args,
            shutdown_event=self.shutdown_event,
        )

    def _completed_futures(
        self,
        futures,
        *,
        wait_label: Optional[str] = None,
        future_owners: Optional[Dict[Any, str]] = None,
    ):
        """Yield completed futures while staying responsive to shutdown."""
        pending = set(futures)
        last_diag = 0.0
        while pending:
            if self.shutdown_requested():
                for fut in pending:
                    fut.cancel()
                break
            done, pending = concurrent.futures.wait(
                pending,
                timeout=0.5,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done and wait_label:
                now = time.monotonic()
                if last_diag <= 0 or now - last_diag >= 30.0:
                    last_diag = now
                    self._diagnostic_log(
                        "    "
                        + self._future_wait_diagnostic_summary(
                            wait_label=wait_label,
                            pending=pending,
                            future_owners=future_owners,
                        )
                    )
            for fut in done:
                yield fut

    def _eval_pool_label(self) -> str:
        if self.eval_executor is not None:
            workers = getattr(self.eval_executor, "max_workers", self.n_eval_workers)
            return f"global workers={workers}"
        return f"pool workers={self.n_eval_workers}"

    @staticmethod
    def _format_counter(counter: Dict[str, Any]) -> str:
        if not counter:
            return "{}"
        return "{" + ", ".join(
            f"{key}={value}" for key, value in sorted(counter.items())
        ) + "}"

    def _executor_diagnostics(self) -> Dict[str, Any]:
        """Return live eval executor diagnostics safe for logs/checkpoints."""
        executor_snapshot: Dict[str, Any] = {}
        if self.eval_executor is not None and hasattr(self.eval_executor, "snapshot"):
            try:
                executor_snapshot = self.eval_executor.snapshot()
            except Exception as exc:
                executor_snapshot = {"snapshot_error": repr(exc)}

        return {
            "eval_owner_id": self.eval_owner_id,
            "active_children": _active_subprocess_count(),
            "phases": _phase_snapshot(),
            "executor": executor_snapshot,
        }

    def _eval_diagnostic_summary(
        self,
        *,
        normalize_futures: int,
        eval_futures: int,
        degen_futures: int = 0,
        raw_remaining: int,
        submitted_total: int,
        prepared_count: int,
        skipped_count: int,
    ) -> str:
        diagnostics = self._executor_diagnostics()
        executor_snapshot = diagnostics.get("executor", {})
        queued_by_fn = executor_snapshot.get("queued_by_fn", {})
        running_by_fn = executor_snapshot.get("running_by_fn", {})
        submitted_by_fn = executor_snapshot.get("submitted_by_fn", {})
        finished_by_fn = executor_snapshot.get("finished_by_fn", {})
        running_by_owner = executor_snapshot.get("running_by_owner", {})

        return (
            "[EvalDiag] "
            f"owner={self.eval_owner_id} "
            f"local_normalize={normalize_futures} local_eval={eval_futures} "
            f"local_degen={degen_futures} "
            f"raw_remaining={raw_remaining} prepared={prepared_count} "
            f"submitted_eval={submitted_total} skipped={skipped_count} "
            f"active_children={diagnostics.get('active_children', 'na')} "
            f"phases={self._format_counter(diagnostics.get('phases', {}))} "
            f"executor_queued={executor_snapshot.get('queued_total', 'na')} "
            f"executor_running={executor_snapshot.get('running_total', 'na')} "
            f"executor_inflight={executor_snapshot.get('inflight_total', 'na')} "
            f"executor_finished={executor_snapshot.get('finished_total', 'na')} "
            f"oldest_queued_s={executor_snapshot.get('oldest_queued_age_s', 'na')} "
            f"oldest_running_s={executor_snapshot.get('oldest_running_age_s', 'na')} "
            f"owner_cap={executor_snapshot.get('owner_max_inflight', 'na')} "
            f"owners={executor_snapshot.get('owners', 'na')} "
            f"ready_owners={executor_snapshot.get('ready_owners', 'na')} "
            f"worker_alive={executor_snapshot.get('worker_processes_alive', 'na')} "
            f"dispatcher_alive={executor_snapshot.get('dispatcher_alive', 'na')} "
            f"result_readers_alive={executor_snapshot.get('result_readers_alive', 'na')} "
            f"result_readers_active={executor_snapshot.get('result_readers_active', 'na')} "
            f"result_seen_age_s={executor_snapshot.get('result_loop_last_seen_age_s', 'na')} "
            f"result_completed_age_s={executor_snapshot.get('result_loop_last_completed_age_s', 'na')} "
            f"complete_task_active={executor_snapshot.get('complete_task_active_count', 'na')} "
            f"worker_exec_started={executor_snapshot.get('worker_exec_started', 'na')} "
            f"worker_exec_finished={executor_snapshot.get('worker_exec_finished', 'na')} "
            f"worker_put_pending={executor_snapshot.get('worker_put_pending', 'na')} "
            f"result_error={executor_snapshot.get('result_loop_error', '')!r} "
            f"queued_by_fn={self._format_counter(queued_by_fn)} "
            f"running_by_fn={self._format_counter(running_by_fn)} "
            f"running_by_owner={self._format_counter(running_by_owner)} "
            f"submitted_by_fn={self._format_counter(submitted_by_fn)} "
            f"finished_by_fn={self._format_counter(finished_by_fn)}"
        )

    def _future_wait_diagnostic_summary(
        self,
        *,
        wait_label: str,
        pending: Set[Any],
        future_owners: Optional[Dict[Any, str]] = None,
    ) -> str:
        diagnostics = self._executor_diagnostics()
        executor_snapshot = diagnostics.get("executor", {})
        queued_by_fn = executor_snapshot.get("queued_by_fn", {})
        running_by_fn = executor_snapshot.get("running_by_fn", {})
        owner_counts: Counter[str] = Counter()
        if future_owners:
            for fut in pending:
                owner = str(future_owners.get(fut, "unknown"))
                if len(owner) > 80:
                    owner = owner
                owner_counts[owner] += 1

        return (
            "[FutureWaitDiag] "
            f"label={wait_label} owner={self.eval_owner_id} "
            f"pending_futures={len(pending)} "
            f"pending_by_owner={self._format_counter(owner_counts)} "
            f"active_children={diagnostics.get('active_children', 'na')} "
            f"phases={self._format_counter(diagnostics.get('phases', {}))} "
            f"executor_queued={executor_snapshot.get('queued_total', 'na')} "
            f"executor_running={executor_snapshot.get('running_total', 'na')} "
            f"executor_inflight={executor_snapshot.get('inflight_total', 'na')} "
            f"oldest_queued_s={executor_snapshot.get('oldest_queued_age_s', 'na')} "
            f"oldest_running_s={executor_snapshot.get('oldest_running_age_s', 'na')} "
            f"queued_by_fn={self._format_counter(queued_by_fn)} "
            f"running_by_fn={self._format_counter(running_by_fn)} "
            f"worker_exec_started={executor_snapshot.get('worker_exec_started', 'na')} "
            f"worker_exec_finished={executor_snapshot.get('worker_exec_finished', 'na')} "
            f"worker_put_pending={executor_snapshot.get('worker_put_pending', 'na')} "
            f"result_error={executor_snapshot.get('result_loop_error', '')!r}"
        )

    def load_checkpoint_state(self, state: Dict[str, Any]) -> None:
        """Restore durable search state from a checkpoint dictionary."""
        search_state = state.get("search_state", {})
        tree_state = search_state.get("tree")
        if tree_state:
            self.tree = EvolutionTree.from_state(tree_state)
        self.param_map = {
            str(k): list(v)
            for k, v in search_state.get("param_map", {}).items()
        }
        self._selection_history = [
            list(step)
            for step in search_state.get("selection_history", [])
        ]
        self._seen_keys = set(search_state.get("seen_keys", []))
        if not self._seen_keys:
            for node in self.tree.all_nodes:
                try:
                    self._seen_keys.add(
                        self.normalizer.structural_key(node.skeleton_str))
                except Exception:
                    pass
        self._preprocessing_rules = list(
            search_state.get("preprocessing_rules", []))
        self.history = list(search_state.get("history", []))
        self._candidate_statuses = {
            str(k): (
                str(v)
                if str(v) in _TERMINAL_CANDIDATE_STATUSES
                or str(v) in _ACTIVE_CANDIDATE_STATUSES
                else "evaluated"
            )
            for k, v in search_state.get("candidate_statuses", {}).items()
        }
        self._active_step = search_state.get("active_step")
        normalized_active_statuses = 0
        for formula, status in list(self._candidate_statuses.items()):
            if status not in {"queued", "running"}:
                continue
            node = self.tree.get_node(formula)
            next_status = (
                "evaluated"
                if node is not None and node.is_evaluated
                else "pending"
            )
            if next_status != status:
                normalized_active_statuses += 1
            self._candidate_statuses[formula] = next_status
        if self._active_step:
            for record in self._active_step.get("candidates", []):
                formula = str(record.get("formula", "")).strip()
                status = str(record.get("status", "pending"))
                if status not in {"queued", "running"}:
                    continue
                node = self.tree.get_node(formula)
                next_status = (
                    "evaluated"
                    if node is not None and node.is_evaluated
                    else "pending"
                )
                if next_status != status:
                    normalized_active_statuses += 1
                record["status"] = next_status
                if formula and formula not in self._candidate_statuses:
                    self._candidate_statuses[formula] = next_status
        final_sweep_state = search_state.get("final_sweep_state")
        self._final_sweep_state = (
            dict(final_sweep_state)
            if isinstance(final_sweep_state, dict)
            else None
        )
        self._mature_select_count = {
            str(k): int(v)
            for k, v in search_state.get("mature_select_count", {}).items()
        }
        self.mature_anneal_budget = max(
            0,
            int(search_state.get(
                "mature_anneal_budget",
                self.mature_anneal_budget,
            )),
        )
        self.mature_train_threshold = float(
            search_state.get(
                "mature_train_threshold",
                self.mature_train_threshold,
            )
        )
        self.mature_test_threshold = float(
            search_state.get(
                "mature_test_threshold",
                self.mature_test_threshold,
            )
        )
        self._seed_initialized = bool(
            search_state.get("seed_initialized", bool(self.tree.all_nodes)))
        self._resume_step = int(
            search_state.get("next_step", len(self.history)))
        self._checkpoint_stage = str(state.get("stage", "resumed"))
        self._log(
            f"[checkpoint] Resumed state: status={state.get('status', 'unknown')}, "
            f"next_step={self._resume_step}, nodes={len(self.tree.all_nodes)}"
        )
        if normalized_active_statuses:
            self._log(
                f"[checkpoint] Normalized {normalized_active_statuses} "
                "checkpoint queued/running candidate statuses for resume."
            )

    def _final_sweep_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        """Return final-sweep state with stable, JSON-friendly counters."""
        if not self._final_sweep_state:
            return None
        state = {
            key: value
            for key, value in _snapshot_items(self._final_sweep_state)
        }
        parent_ids = [str(pid) for pid in state.get("parent_ids", [])]
        completed = [str(pid) for pid in state.get("completed_parent_ids", [])]
        generated = [str(pid) for pid in state.get("generated_parent_ids", [])]
        skipped = [str(pid) for pid in state.get("skipped_parent_ids", [])]
        parent_set = set(parent_ids)
        state["parent_ids"] = [pid for pid in parent_ids if pid in parent_set]
        state["completed_parent_ids"] = [
            pid for pid in completed if pid in parent_set
        ]
        state["generated_parent_ids"] = [
            pid for pid in generated if pid in parent_set
        ]
        state["skipped_parent_ids"] = [
            pid for pid in skipped if pid in parent_set
        ]
        state["n_parents"] = len(parent_ids)
        state["n_completed_parents"] = len(set(state["completed_parent_ids"]))
        state["n_generated_parents"] = len(set(state["generated_parent_ids"]))
        state["n_skipped_parents"] = len(set(state["skipped_parent_ids"]))
        return state

    def _active_step_state(
        self,
        candidate_statuses: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return active step state with fresh per-candidate statuses."""
        if not self._active_step:
            return None
        active = {
            key: value
            for key, value in _snapshot_items(self._active_step)
        }
        statuses = candidate_statuses
        if statuses is None:
            statuses = {
                str(formula): str(status)
                for formula, status in _snapshot_items(
                    self._candidate_statuses)
            }
        candidates = []
        for record in _snapshot_list(active.get("candidates", [])):
            item = dict(record)
            formula = str(item.get("formula", ""))
            node = self.tree.get_node(formula)
            if node is not None and node.is_evaluated:
                status = statuses.get(formula, "evaluated")
            else:
                status = statuses.get(
                    formula, item.get("status", "pending"))
            item["status"] = status
            candidates.append(item)
        active["candidates"] = candidates
        active["n_candidates"] = len(candidates)
        n_evaluated = 0
        n_pending = 0
        for item in candidates:
            status = item.get("status", "pending")
            node = self.tree.get_node(str(item.get("formula", "")))
            if status == "evaluated" or (node is not None and node.is_evaluated):
                n_evaluated += 1
            if status not in _TERMINAL_CANDIDATE_STATUSES:
                n_pending += 1
        active["n_evaluated"] = n_evaluated
        active["n_pending"] = n_pending
        return active

    def checkpoint_state(
        self,
        status: str = "running",
        stage: Optional[str] = None,
        next_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build a JSON-serializable durable checkpoint state."""
        if stage is not None:
            self._checkpoint_stage = stage
        if next_step is None:
            next_step = self._resume_step
        best = self.tree.best_node
        metadata = dict(self.checkpoint_metadata)
        metadata.setdefault("equation", self.dataset.equation_name)
        tree_state = self.tree.to_state(evaluated_only=False)
        durable_formulas = {
            str(node_state.get("skeleton_str", ""))
            for node_state in tree_state.get("nodes", [])
            if node_state.get("skeleton_str")
        }
        param_map = {
            str(formula): list(params)
            for formula, params in _snapshot_items(self.param_map)
            if str(formula) in durable_formulas
        }
        candidate_statuses = {
            str(formula): str(candidate_status)
            for formula, candidate_status in _snapshot_items(
                self._candidate_statuses)
        }
        mature_select_count = {
            str(formula): count
            for formula, count in _snapshot_items(self._mature_select_count)
            if str(formula) in durable_formulas
        }
        selection_history = [
            list(step)
            for step in _snapshot_list(self._selection_history)
        ]
        seen_keys = sorted(_snapshot_list(self._seen_keys))
        preprocessing_rules = _snapshot_list(self._preprocessing_rules)
        history = _snapshot_list(self.history)
        active_step = self._active_step_state(candidate_statuses)
        final_sweep_state = self._final_sweep_checkpoint_state()
        diagnostics = self._executor_diagnostics()
        return {
            "schema_version": 2,
            "updated_at": utc_timestamp(),
            "status": status,
            "stage": self._checkpoint_stage,
            **metadata,
            "diagnostics": diagnostics,
            "search_state": {
                "tree": tree_state,
                "param_map": param_map,
                "selection_history": selection_history,
                "seen_keys": seen_keys,
                "candidate_statuses": candidate_statuses,
                "active_step": active_step,
                "final_sweep_state": final_sweep_state,
                "preprocessing_rules": preprocessing_rules,
                "history": history,
                "mature_select_count": mature_select_count,
                "mature_anneal_budget": self.mature_anneal_budget,
                "mature_train_threshold": self.mature_train_threshold,
                "mature_test_threshold": self.mature_test_threshold,
                "seed_initialized": self._seed_initialized,
                "next_step": next_step,
            },
            "best": {
                "expression": best.skeleton_str if best else None,
                "train_nmse": best.train_nmse if best else None,
                "test_nmse": best.test_nmse if best else None,
                "ood_test_nmse": best.ood_test_nmse if best else None,
            },
        }

    def save_checkpoint(
        self,
        status: str = "running",
        stage: Optional[str] = None,
        next_step: Optional[int] = None,
        force: bool = False,
    ) -> None:
        """Persist durable state, throttled per equation thread."""
        if not self.checkpoint_path:
            return
        if stage is not None:
            self._checkpoint_stage = stage

        now = time.monotonic()
        force_write = force or status != "running"
        if (
            not force_write
            and self._last_checkpoint_monotonic > 0
            and self.checkpoint_interval > 0
            and now - self._last_checkpoint_monotonic < self.checkpoint_interval
        ):
            return

        with self._lock:
            state = self.checkpoint_state(
                status=status, stage=stage, next_step=next_step)
            write_checkpoint_atomic(self.checkpoint_path, state)
            self._last_checkpoint_monotonic = now
            try:
                eq_tag = str(
                    state.get("equation_tag")
                    or state.get("equation")
                    or self.dataset.equation_name
                    or "unknown"
                ).replace("/", "_")
                update_manifest(
                    Path(self.checkpoint_path).parent,
                    eq_tag,
                    {
                        "status": status,
                        "stage": state.get("stage"),
                        "checkpoint": Path(self.checkpoint_path).name,
                        "updated_at": state.get("updated_at"),
                        "dataset": state.get("dataset"),
                        "equation": state.get("equation"),
                        "next_step": state["search_state"].get("next_step"),
                        "best": state.get("best", {}),
                    },
                )
            except Exception:
                pass

    def _candidate_record(
        self,
        item: Tuple[str, List[str], str, EvolutionNode, EvolutionNode, str],
    ) -> Dict[str, Any]:
        formula, params, mutation, child_node, parent_node, tag = item
        return {
            "formula": formula,
            "params": list(params),
            "mutation": mutation,
            "parent_id": parent_node.node_id,
            "parent_formula": parent_node.skeleton_str,
            "node_parent_id": self.tree.get_node_id(child_node.parent_id),
            "node_parent_formula": child_node.parent_id,
            "tag": tag,
            "status": self._candidate_statuses.get(formula, "pending"),
        }

    def _record_parent_formula(self, record: Dict[str, Any]) -> str:
        """Resolve a checkpoint candidate parent to the internal formula key.

        New records store numeric `parent_id` plus `parent_formula`; older
        checkpoints stored the formula directly in `parent_id`.
        """
        for key in ("parent_formula", "node_parent_formula"):
            formula = str(record.get(key, "")).strip()
            if formula and self.tree.get_node(formula) is not None:
                return formula

        for key in ("parent_id", "node_parent_id"):
            value = record.get(key)
            if value is None:
                continue
            formula = str(value).strip()
            if formula and self.tree.get_node(formula) is not None:
                return formula
            resolved = self.tree.resolve_formula_ref(value)
            if resolved:
                return resolved

        return ""

    def _candidate_already_claimed(
        self,
        formula: str,
        node: Optional[EvolutionNode] = None,
    ) -> bool:
        """Return True if a formula is already pending, running, or final."""
        status = self._candidate_statuses.get(formula)
        if node is not None and node.is_evaluated:
            return True
        return (
            status in _ACTIVE_CANDIDATE_STATUSES
            or status in _TERMINAL_CANDIDATE_STATUSES
        )

    def _find_existing_equivalent_node(
        self,
        formula: str,
        canonical_key: Optional[str] = None,
    ) -> Optional[EvolutionNode]:
        existing = self.tree.get_node(formula)
        if existing is not None:
            return existing
        key = canonical_key
        if not key:
            try:
                key = self.normalizer.structural_key(formula)
            except Exception:
                return None
        for node in self.tree.all_nodes:
            node_key = node.canonical_key
            if not node_key:
                if canonical_key:
                    continue
                try:
                    node_key = self.normalizer.structural_key(
                        node.skeleton_str)
                    node.canonical_key = node_key
                except Exception:
                    continue
            if node_key == key:
                return node
        return None

    def _claim_candidate_node(
        self,
        formula: str,
        params: List[str],
        parent_formula: Optional[str],
        allow_active_reclaim: bool = False,
        canonical_key: Optional[str] = None,
    ) -> Tuple[Optional[EvolutionNode], List[str], bool]:
        """Create or claim a candidate node exactly once.

        Returns ``(node, params, skipped_for_params)``. A ``None`` node means
        the candidate is a duplicate or otherwise should not be submitted.
        """
        formula = formula.strip()
        params = list(params)
        if not formula:
            return None, params, False

        existing_node = self.tree.get_node(formula)
        if existing_node is not None:
            if canonical_key and not existing_node.canonical_key:
                existing_node.canonical_key = canonical_key
            if parent_formula:
                self.tree.attach_child(parent_formula, formula)
            status = self._candidate_statuses.get(formula)
            is_active = status in _ACTIVE_CANDIDATE_STATUSES
            is_terminal = status in _TERMINAL_CANDIDATE_STATUSES
            if existing_node.is_evaluated or is_terminal:
                return None, params, False
            if is_active and not allow_active_reclaim:
                return None, params, False
            params = self.param_map.get(formula, params)
            if len(params) > self.max_params:
                return None, params, True
            self._candidate_statuses[formula] = "pending"
            return existing_node, params, False

        if len(params) > self.max_params:
            return None, params, True

        key = canonical_key or self.normalizer.structural_key(formula)
        if key in self._seen_keys:
            existing_equiv = self._find_existing_equivalent_node(
                formula, canonical_key=key)
            if existing_equiv is not None and parent_formula:
                self.tree.attach_child(
                    parent_formula, existing_equiv.skeleton_str)
            return None, params, False
        self._seen_keys.add(key)

        node = self.tree.add_node(formula, parent_formula=parent_formula)
        node.canonical_key = key
        self.param_map[formula] = params
        self._candidate_statuses[formula] = "pending"
        return node, params, False

    def _record_active_candidates(
        self,
        *,
        step: int,
        phase: str,
        parent_ids: List[str],
        pending: List[
            Tuple[str, List[str], str, EvolutionNode, EvolutionNode, str]
        ],
    ) -> None:
        """Persist the current in-step evaluation batch for precise resume."""
        if not pending:
            return
        if not self._active_step or int(self._active_step.get("step", -1)) != step:
            self._active_step = {
                "step": step,
                "phase": phase,
                "parent_ids": list(parent_ids),
                "candidates": [],
            }
        else:
            self._active_step["phase"] = phase
            existing_parents = list(self._active_step.get("parent_ids", []))
            for parent_id in parent_ids:
                if parent_id not in existing_parents:
                    existing_parents.append(parent_id)
            self._active_step["parent_ids"] = existing_parents

        records = list(self._active_step.get("candidates", []))
        seen_formulas = {str(item.get("formula", "")) for item in records}
        for item in pending:
            formula = item[0]
            self._candidate_statuses.setdefault(formula, "pending")
            if formula not in seen_formulas:
                records.append(self._candidate_record(item))
                seen_formulas.add(formula)
        self._active_step["candidates"] = records

    def _active_step_matches(self, step: int) -> bool:
        return bool(
            self._active_step
            and int(self._active_step.get("step", -1)) == step
            and self._active_step.get("candidates")
        )

    def _parent_cands_from_active_step(
        self,
    ) -> List[Tuple[str, EvolutionNode, List[Dict[str, Any]]]]:
        """Rebuild pending parent/candidate groups from checkpoint state."""
        if not self._active_step:
            return []
        grouped: Dict[str, Tuple[EvolutionNode, List[Dict[str, Any]]]] = {}
        for record in self._active_step.get("candidates", []):
            formula = str(record.get("formula", "")).strip()
            if not formula:
                continue
            node = self.tree.get_node(formula)
            status = self._candidate_statuses.get(
                formula, record.get("status", "pending"))
            if node is not None and node.is_evaluated:
                if status not in _TERMINAL_CANDIDATE_STATUSES:
                    self._candidate_statuses[formula] = "evaluated"
                continue
            if status in _TERMINAL_CANDIDATE_STATUSES:
                continue

            parent_id = self._record_parent_formula(record)
            parent_node = self.tree.get_node(parent_id)
            if parent_node is None:
                continue
            grouped.setdefault(parent_id, (parent_node, []))[1].append({
                "expression": formula,
                "params": list(record.get("params", [])),
                "mutation": str(record.get("mutation", "[checkpoint]")),
                "_resume_active": True,
            })

        return [
            (parent_id, parent_node, candidates)
            for parent_id, (parent_node, candidates) in grouped.items()
            if candidates
        ]

    def _results_from_active_step(
        self,
    ) -> List[Tuple[str, EvalResult, EvolutionNode, str]]:
        """Recover already evaluated results for the active step."""
        if not self._active_step:
            return []
        results: List[Tuple[str, EvalResult, EvolutionNode, str]] = []
        seen: Set[str] = set()
        for record in self._active_step.get("candidates", []):
            formula = str(record.get("formula", ""))
            if not formula or formula in seen:
                continue
            node = self.tree.get_node(formula)
            if node is None or not node.is_evaluated:
                continue
            seen.add(formula)
            results.append((
                formula,
                EvalResult(
                    None,
                    list(node.fitted_params),
                    node.train_nmse,
                    node.test_nmse,
                    node.ood_test_nmse,
                ),
                node,
                str(record.get("mutation", "[checkpoint]")),
            ))
        return results

    def shutdown_requested(self) -> bool:
        return bool(self.shutdown_event and self.shutdown_event.is_set())

    # ------------------------------------------------------------------ #
    # Initialize population
    # ------------------------------------------------------------------ #

    def initialize_seeds(
        self, seed_skeletons: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Initialize evolution tree root nodes."""
        if self._seed_initialized and self.tree.all_nodes:
            self._log("[checkpoint] Initial seed population already restored; skipping initialization.")
            return

        self.save_checkpoint(stage="initializing_seeds", next_step=0)
        self._log("Generating domain knowledge...")
        dk, preproc_rules = self.generator.generate_domain_knowledge(
            context_prompt=self.dataset.get_context_prompt(),
            variables=self.dataset.feature_names,
        )
        self._preprocessing_rules = preproc_rules or []

        # Sync domain knowledge to LLM mutator
        if hasattr(self.llm_mutator, 'domain_knowledge'):
            self.llm_mutator.domain_knowledge = dk or ""

        if dk:
            self._log(f"Domain knowledge ready:\n{dk}")
        else:
            self._log("Domain knowledge generation returned no result; using generic priors.")
        if self._preprocessing_rules:
            type_desc = {"linear_scale": "linear scale", "moment": "compute moment"}
            for r in self._preprocessing_rules:
                self._log(
                    f"  Preprocessing rule: {r['variable']} -> "
                    f"{type_desc.get(r['type'], r['type'])} "
                    f"({r.get('reason', '')})"
                )

        if seed_skeletons is None:
            self._log("Calling LLM to generate initial seed formulas...")
            seed_skeletons = self.generator.initialize_seeds(
                context_prompt=self.dataset.get_context_prompt(),
                variables=self.dataset.feature_names,
                n_seeds=self.n_seeds,
            )

        self._log(f"Initializing {len(seed_skeletons)} seed nodes...\n")

        raw_seed_items: list = []
        if self.tree.all_nodes:
            for node in self.tree.all_nodes:
                status = self._candidate_statuses.get(
                    node.skeleton_str, "pending")
                if (
                    node.parent_id is None
                    and not node.is_evaluated
                    and status not in _TERMINAL_CANDIDATE_STATUSES
                ):
                    raw_seed_items.append((
                        node.skeleton_str,
                        self.param_map.get(node.skeleton_str, []),
                        node,
                        "[Seed]",
                        True,
                    ))
        else:
            for seed in seed_skeletons:
                formula = seed.get("expression", "").strip()
                params = seed.get("params", [])
                if not formula:
                    continue
                raw_seed_items.append((
                    formula,
                    list(params),
                    None,
                    "[Seed]",
                    False,
                ))

        if raw_seed_items:
            self.save_checkpoint(stage="seed_candidates_pending", next_step=0)

        evaluated_seeds: list = []
        if raw_seed_items:
            if self.eval_executor is None:
                pool_cm = concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.n_eval_workers)
                pool = pool_cm.__enter__()
                close_pool = True
            else:
                pool = self.eval_executor
                pool_cm = None
                close_pool = False
            try:
                normalize_futures: Dict[Any, Tuple[
                    str,
                    List[str],
                    Optional[EvolutionNode],
                    str,
                    bool,
                ]] = {}
                eval_futures: Dict[Any, Tuple[str, List[str], EvolutionNode, str]] = {}
                degen_futures: Dict[Any, Tuple[
                    str,
                    EvolutionNode,
                    str,
                    str,
                    EvalResult,
                ]] = {}
                submitted_total = 0
                initial_submitted = 0
                normalize_timeout = self.normalize_timeout

                def _submit_seed_normalize(
                    raw_item: Tuple[
                        str,
                        List[str],
                        Optional[EvolutionNode],
                        str,
                        bool,
                    ],
                ) -> None:
                    raw_formula, raw_params, *_ = raw_item
                    fut = self._submit_normalization(
                        pool,
                        self.dataset.feature_names,
                        raw_formula,
                        raw_params,
                        False,
                        normalize_timeout,
                    )
                    normalize_futures[fut] = raw_item

                def _submit_seed_eval(
                    formula: str,
                    params: List[str],
                    node: EvolutionNode,
                    sp_expr: sp.Expr,
                    mutation: str = "[Seed]",
                ) -> None:
                    self._candidate_statuses[formula] = "queued"
                    fut = self._submit_candidate_eval(
                        pool,
                        self.evaluator,
                        sp_expr,
                        formula,
                        params,
                        None,
                        self.timeout,
                    )
                    eval_futures[fut] = (formula, params, node, mutation)

                def _claim_and_submit_seed_normalized(future: Any) -> None:
                    nonlocal submitted_total, initial_submitted
                    raw_formula, raw_params, existing_node, mutation, allow_active_reclaim = (
                        normalize_futures.pop(future)
                    )
                    try:
                        norm_result = future.result()
                    except Exception as e:
                        self._log(
                            f"    ✗ {mutation} normalization exception: "
                            f"{raw_formula} ({e})"
                        )
                        return

                    formula, sp_expr_prepared, params, canonical_key, norm_status = norm_result
                    if norm_status not in {"ok", "timeout_raw", "timeout_softkey"}:
                        self._debug_log(
                            f"    ✗ {mutation} normalization {norm_status}: "
                            f"{raw_formula}"
                        )
                        return
                    if norm_status == "timeout_softkey":
                        self._log(
                            f"    ⚠ {mutation} normalization timeout; evaluating raw "
                            f"expression (dedup key kept): {raw_formula}"
                        )
                    elif norm_status == "timeout_raw":
                        self._log(
                            f"    ⚠ {mutation} normalization timeout; "
                            f"evaluating raw expression: {raw_formula}"
                        )
                    if not formula or sp_expr_prepared is None:
                        return

                    if (
                        existing_node is not None
                        and existing_node.skeleton_str == formula
                    ):
                        node = existing_node
                        if canonical_key and not node.canonical_key:
                            node.canonical_key = canonical_key
                        params = self.param_map.get(formula, list(params))
                        if len(params) > self.max_params:
                            return
                        self._candidate_statuses[formula] = "pending"
                    else:
                        node, params, skipped_params = self._claim_candidate_node(
                            formula,
                            list(params),
                            parent_formula=None,
                            allow_active_reclaim=allow_active_reclaim,
                            canonical_key=canonical_key,
                        )
                        if skipped_params or node is None:
                            return

                    self.save_checkpoint(
                        stage="seed_candidates_pending",
                        next_step=0,
                    )
                    _submit_seed_eval(
                        formula,
                        list(params),
                        node,
                        sp_expr_prepared,
                        mutation,
                    )
                    submitted_total += 1
                    initial_submitted += 1

                def _enqueue_simplified_seed(
                    parent_node: EvolutionNode,
                    expression: Optional[str],
                    params: Optional[List[str]],
                    canonical_key: Optional[str],
                    sp_expr: Optional[sp.Expr] = None,
                ) -> bool:
                    formula = str(expression or "").strip()
                    simp_params = list(params or [])
                    if not formula or len(simp_params) > self.max_params:
                        return False

                    # O(1) fast-path before taking the lock
                    key_for_check = canonical_key or formula
                    if key_for_check in self._seen_keys:
                        return False

                    node, simp_params, _ = self._claim_candidate_node(
                        formula,
                        simp_params,
                        parent_formula=parent_node.skeleton_str,
                        canonical_key=canonical_key,
                    )
                    if node is None:
                        return False
                    resolved_sp_expr = sp_expr if sp_expr is not None else self.normalizer.parse(formula)
                    if resolved_sp_expr is None:
                        return False
                    self._candidate_statuses[formula] = "queued"
                    fut = self._submit_candidate_eval(
                        pool,
                        self.evaluator,
                        resolved_sp_expr,
                        formula,
                        simp_params,
                        None,
                        self.timeout,
                    )
                    eval_futures[fut] = (formula, simp_params, node, "[Seed simplified]")
                    self.save_checkpoint(
                        stage="seed_simplified_pending",
                        next_step=0,
                    )
                    return True

                for raw_item in raw_seed_items:
                    _submit_seed_normalize(raw_item)

                def _submit_seed_degen(
                    formula: str,
                    node: EvolutionNode,
                    mutation: str,
                    result_formula: str,
                    result_param_names: List[str],
                    result_key: str,
                    result: EvalResult,
                    norm_params: List[float],
                ) -> None:
                    fut = self._submit_degeneracy_check(
                        pool,
                        self.evaluator,
                        self.mutator,
                        result_formula,
                        result_param_names,
                        norm_params,
                        self.degeneracy_timeout,
                    )
                    degen_futures[fut] = (
                        formula,
                        node,
                        mutation,
                        result_key,
                        result,
                    )

                while normalize_futures or eval_futures or degen_futures:
                    if self.shutdown_requested():
                        for fut in normalize_futures:
                            fut.cancel()
                        for fut in eval_futures:
                            fut.cancel()
                        for fut in degen_futures:
                            fut.cancel()
                        break
                    wait_set = (
                        set(normalize_futures)
                        | set(eval_futures)
                        | set(degen_futures)
                    )
                    done, _ = concurrent.futures.wait(
                        wait_set,
                        timeout=0.5,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for future in done:
                        if future in normalize_futures:
                            _claim_and_submit_seed_normalized(future)
                            continue

                        if future in eval_futures:
                            formula, params, node, mutation = eval_futures.pop(future)
                            self._candidate_statuses[formula] = "running"
                            try:
                                (
                                    result_formula,
                                    result_param_names,
                                    result_key,
                                    result,
                                    norm_params,
                                ) = future.result()
                            except Exception as e:
                                self._candidate_statuses[formula] = "exception"
                                self._log(
                                    f"    ✗ {mutation} evaluation exception: "
                                    f"{formula} ({e})"
                                )
                                self.save_checkpoint(
                                    stage="seed_candidate_exception",
                                    next_step=0,
                                )
                                continue

                            if result.train_nmse == float("inf"):
                                self._candidate_statuses[formula] = "failed"
                                reason = (
                                    getattr(result, "fail_reason", "") or "unknown"
                                )
                                self._debug_log(
                                    f"    ✗ {mutation} evaluation rejected ({reason}): "
                                    f"{formula}"
                                )
                                self.save_checkpoint(
                                    stage="seed_candidate_failed",
                                    next_step=0,
                                )
                                continue

                            self.tree.update_score(
                                result_formula,
                                train_nmse=result.train_nmse,
                                test_nmse=result.test_nmse,
                                ood_test_nmse=result.ood_test_nmse,
                                param_names=result_param_names,
                                fitted_params=norm_params,
                            )
                            _submit_seed_degen(
                                formula,
                                node,
                                mutation,
                                result_formula,
                                result_param_names,
                                result_key,
                                result,
                                norm_params,
                            )
                            continue

                        (
                            formula,
                            node,
                            mutation,
                            result_key,
                            result,
                        ) = degen_futures.pop(future)
                        simp_sp_expr_seed: Optional[sp.Expr] = None
                        try:
                            (
                                degen_status,
                                simp_expr,
                                simp_params,
                                simp_key,
                                degen_reasons,
                                simp_sp_expr_seed,
                            ) = future.result()
                        except Exception as e:
                            degen_status, simp_expr, simp_params, simp_key = (
                                "ok", None, None, "")
                            degen_reasons = [f"degen_exception:{e}"]

                        if degen_status == "timeout":
                            self._log(
                                f"    ⚠ {mutation} degeneracy check timeout "
                                f"(>{self.degeneracy_timeout:.0f}s): "
                                f"{formula}"
                            )
                            degen_status = "ok"
                        elif degen_status == "interrupted":
                            self._candidate_statuses[formula] = "interrupted"
                        elif degen_status == "overfit":
                            node.mark_degenerated(
                                None,
                                degen_reasons,
                                kind="overfit",
                            )
                            self._candidate_statuses[formula] = "evaluated"
                            if degen_reasons:
                                self._log(
                                    "    ↳ Seed degenerated (overfitting): "
                                    f"{'; '.join(degen_reasons)}"
                                )
                        elif degen_status == "simplified":
                            if simp_expr:
                                node.canonical_key = simp_key
                                node.add_degeneration_candidate(
                                    simp_expr,
                                    degen_reasons,
                                    kind="simplified",
                                )
                            self._candidate_statuses[formula] = "evaluated"
                            if degen_reasons:
                                self._log(
                                    "    ↳ Seed degenerated (simplified): "
                                    f"{'; '.join(degen_reasons)}"
                                )
                            if _enqueue_simplified_seed(
                                node, simp_expr, simp_params, simp_key,
                                simp_sp_expr_seed):
                                submitted_total += 1
                        else:
                            node.canonical_key = result_key
                            self._candidate_statuses[formula] = "evaluated"

                        evaluated_seeds.append((formula, node, result))
                        self.save_checkpoint(
                            stage="seed_candidate_evaluated",
                            next_step=0,
                        )

                if submitted_total > initial_submitted:
                    self._log(
                        f"  [Seed simplification] Submitted "
                        f"{submitted_total - initial_submitted} simplified seeds inline"
                    )
            finally:
                if close_pool and pool_cm is not None:
                    if self.shutdown_requested():
                        pool.shutdown(wait=False, cancel_futures=True)
                    else:
                        pool_cm.__exit__(None, None, None)

        valid_seeds = [
            (f, n, r) for f, n, r in evaluated_seeds if not n.is_degenerated
        ]
        if self.shutdown_requested():
            self.save_checkpoint(
                status="interrupted",
                stage="interrupted_during_seed_evaluation",
                next_step=0,
            )

        if self.enable_describe and valid_seeds and not self.shutdown_requested():
            desc_items = [
                {
                    "formula": formula,
                    "fitted_params_str": node.get_fitted_params_str(),
                    "mse": result.train_nmse,
                }
                for formula, node, result in valid_seeds
            ]
            descriptions = self.generator.describe_batch(desc_items)
            for (formula, node, result), desc in zip(valid_seeds, descriptions):
                if desc:
                    node.description = desc

        for formula, node, result in evaluated_seeds:
            degen_tag = " [Degenerated]" if node.is_degenerated else ""
            self._log(
                f"  [Seed]{degen_tag} {formula}\n"
                f"         params: {node.get_fitted_params_str() or 'no constant params'}\n"
                f"         description: {node.description}\n"
                f"         train={self._fmt(result.train_nmse)}"
                f"  test={self._fmt(result.test_nmse)}"
                f"  ood={self._fmt(result.ood_test_nmse)}"
            )

        self._mark_mature_nodes()
        self._seed_initialized = True
        self._resume_step = 0
        self._log(f"\nEvolution tree initialization complete: {self.tree}\n")
        self.save_checkpoint(stage="seeds_initialized", next_step=0)

    # ------------------------------------------------------------------ #
    # Final sweep resume helpers
    # ------------------------------------------------------------------ #

    def _final_sweep_add_ids(self, key: str, parent_ids: List[str]) -> None:
        if not self._final_sweep_state:
            return
        existing = list(self._final_sweep_state.get(key, []))
        seen = set(existing)
        for parent_id in parent_ids:
            if parent_id and parent_id not in seen:
                existing.append(parent_id)
                seen.add(parent_id)
        self._final_sweep_state[key] = existing

    def _has_mature_ancestor(self, node: EvolutionNode) -> bool:
        """Return True when an earlier node in this lineage is already mature."""
        parent_id = node.parent_id
        seen: Set[str] = set()
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = self.tree.get_node(parent_id)
            if parent is None:
                return False
            if (
                parent.skeleton_str in self._current_mature_formulas
                and not parent.is_degenerated
            ):
                return True
            parent_id = parent.parent_id
        return False

    def _is_final_sweep_eligible_parent_node(self, node: EvolutionNode) -> bool:
        return not self._has_mature_ancestor(node)

    def _final_sweep_eligible_parent_ids(self) -> Set[str]:
        """Parents eligible for final sweep: first mature node on each lineage.

        Final-sweep children can themselves meet the mature thresholds. Those
        refined descendants should remain in the tree as results, but must not
        become final-sweep parents again after a restart or state rebuild.
        """
        eligible: Set[str] = set()
        self._mark_mature_nodes(log_new=False)
        for node in self.tree.all_nodes:
            if (
                node.skeleton_str in self._current_mature_formulas
                and not node.is_degenerated
                and self._is_final_sweep_eligible_parent_node(node)
            ):
                eligible.add(node.skeleton_str)
        return eligible

    def _prune_final_sweep_state(self) -> None:
        if not self._final_sweep_state:
            return

        eligible = self._final_sweep_eligible_parent_ids()
        parent_ids = list(self._final_sweep_state.get("parent_ids", []))
        pruned_parent_ids = [pid for pid in parent_ids if pid in eligible]
        if len(pruned_parent_ids) == len(parent_ids):
            return

        removed = len(parent_ids) - len(pruned_parent_ids)
        keep = set(pruned_parent_ids)
        self._final_sweep_state["parent_ids"] = pruned_parent_ids
        for key in (
            "completed_parent_ids",
            "generated_parent_ids",
            "skipped_parent_ids",
        ):
            self._final_sweep_state[key] = [
                pid for pid in self._final_sweep_state.get(key, [])
                if pid in keep
            ]
        self._final_sweep_state["lineage_filter"] = "first_mature_only"
        self._log(
            f"[checkpoint] Pruned {removed} final-sweep mature descendants; "
            f"{len(pruned_parent_ids)} first-mature parents remain."
        )

        if (
            self._active_step
            and self._active_step.get("phase") == "final_sweep"
        ):
            records = [
                record for record in self._active_step.get("candidates", [])
                if self._record_parent_formula(record) in keep
            ]
            if records:
                self._active_step["candidates"] = records
                self._active_step["n_candidates"] = len(records)
                self._active_step["n_pending"] = sum(
                    1 for record in records
                    if self._candidate_statuses.get(
                        str(record.get("formula", "")),
                        record.get("status", "pending"),
                    ) not in _TERMINAL_CANDIDATE_STATUSES
                )
                self._active_step["n_evaluated"] = (
                    len(records) - self._active_step["n_pending"]
                )
                self._active_step["parent_ids"] = [
                    pid for pid in self._active_step.get("parent_ids", [])
                    if pid in keep
                ]
            else:
                self._active_step = None

    def _mark_final_sweep_completed_from_active_step(self) -> List[str]:
        """Mark final-sweep parents complete when all recorded candidates ended."""
        if (
            not self._final_sweep_state
            or not self._active_step
            or self._active_step.get("phase") != "final_sweep"
        ):
            return []

        records_by_parent: Dict[str, List[Dict[str, Any]]] = {}
        for record in self._active_step.get("candidates", []):
            parent_id = self._record_parent_formula(record)
            if parent_id:
                records_by_parent.setdefault(parent_id, []).append(record)

        completed: List[str] = []
        for parent_id, records in records_by_parent.items():
            if not records:
                continue
            all_terminal = True
            for record in records:
                formula = str(record.get("formula", ""))
                node = self.tree.get_node(formula)
                status = self._candidate_statuses.get(
                    formula, record.get("status", "pending"))
                if node is not None and node.is_evaluated:
                    if status not in _TERMINAL_CANDIDATE_STATUSES:
                        self._candidate_statuses[formula] = "evaluated"
                    continue
                if status not in _TERMINAL_CANDIDATE_STATUSES:
                    all_terminal = False
                    break
            if all_terminal:
                completed.append(parent_id)

        if completed:
            self._final_sweep_add_ids("completed_parent_ids", completed)
        return completed

    def _run_final_sweep(self) -> None:
        """Run or resume the final parent sweep used by refine_output."""
        if not self.refine_output or self.shutdown_requested():
            return
        if (
            self._final_sweep_state
            and self._final_sweep_state.get("active") is False
        ):
            self._log("[checkpoint] Final sweep already completed; skipping.")
            return

        sweep_results: List[Tuple[str, EvalResult, EvolutionNode, str]] = []

        if self._final_sweep_state and self._final_sweep_state.get("active"):
            self._prune_final_sweep_state()
            parent_ids = list(self._final_sweep_state.get("parent_ids", []))
            completed_ids = set(
                self._final_sweep_state.get("completed_parent_ids", [])
            )
            self._log(
                f"\n[checkpoint] Resuming final sweep: "
                f"{len(completed_ids)}/{len(parent_ids)} parents completed"
            )
        else:
            all_ever_selected: Set[str] = set()
            for step_parents in self._selection_history:
                all_ever_selected.update(step_parents)

            mature_nodes = self._mark_mature_nodes(log_new=False)
            mature_nodes = [
                n for n in mature_nodes
                if self._is_final_sweep_eligible_parent_node(n)
            ]

            supplement: List[EvolutionNode] = []
            if len(mature_nodes) < self.max_mature_nodes:
                supplement = sorted(
                    [n for n in self.tree.all_nodes
                     if n.is_evaluated
                     and n.skeleton_str not in self._current_mature_formulas
                     and not n.is_degenerated
                     and self._is_final_sweep_eligible_parent_node(n)],
                    key=lambda n: n.train_nmse,
                )[:self.max_mature_nodes - len(mature_nodes)]

            must_be_parents = mature_nodes + supplement
            unselected = [
                n for n in must_be_parents
                if n.skeleton_str not in all_ever_selected
            ]
            if not unselected:
                return

            parent_ids = [n.skeleton_str for n in unselected]
            self._final_sweep_state = {
                "active": True,
                "parent_ids": parent_ids,
                "completed_parent_ids": [],
                "generated_parent_ids": [],
                "skipped_parent_ids": [],
                "mature_count": len(mature_nodes),
                "supplement_count": len(supplement),
                "lineage_filter": "first_mature_only",
            }
            self.save_checkpoint(
                stage="final_sweep_started",
                next_step=self._resume_step,
                force=True,
            )

            self._log(f"\n{'='*60}")
            self._log(
                f"[Final sweep] Mature {len(mature_nodes)} + "
                f"supplement {len(supplement)} = "
                f"total {len(must_be_parents)} candidates, "
                f"{len(unselected)} not yet selected as parent"
            )
            self._log("=" * 60)

        if (
            self._active_step
            and self._active_step.get("phase") == "final_sweep"
        ):
            recovered = self._results_from_active_step()
            parent_cands = self._parent_cands_from_active_step()
            self._log(
                f"[checkpoint] Resuming final sweep eval: "
                f"{len(recovered)} evaluated, "
                f"{sum(len(cands) for _, _, cands in parent_cands)} pending"
            )
            sweep_results.extend(recovered)
            if parent_cands:
                parent_ids_used = [
                    parent_id for parent_id, _, _ in parent_cands
                ]
                new_results = self._evaluate_all_parents(
                    parent_cands,
                    phase="final_sweep",
                    parent_ids=parent_ids_used,
                )
                sweep_results.extend(new_results)

            pending_after_resume = self._parent_cands_from_active_step()
            self._mark_final_sweep_completed_from_active_step()
            if not pending_after_resume:
                self._active_step = None
            self.save_checkpoint(
                stage="final_sweep_eval_resumed",
                next_step=self._resume_step,
                force=True,
            )

        if self.shutdown_requested():
            self.save_checkpoint(
                status="interrupted",
                stage="interrupted_during_final_sweep",
                next_step=self._resume_step,
                force=True,
            )
            return

        parent_ids = list((self._final_sweep_state or {}).get("parent_ids", []))
        completed_ids = set(
            (self._final_sweep_state or {}).get("completed_parent_ids", [])
        )
        parent_positions = {
            parent_id: i for i, parent_id in enumerate(parent_ids)
        }
        remaining_nodes = [
            self.tree.get_node(parent_id)
            for parent_id in parent_ids
            if parent_id not in completed_ids
        ]
        remaining_nodes = [node for node in remaining_nodes if node is not None]

        if remaining_nodes:
            n_sweep_workers = min(self.n_parent_workers, len(remaining_nodes))
            eval_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, n_sweep_workers))
            eval_futs: Dict[concurrent.futures.Future, str] = {}

            def _sweep_one(node: EvolutionNode):
                if self.shutdown_requested():
                    return None
                parent_id = node.skeleton_str
                pi = parent_positions.get(parent_id, 0)
                mature_tag = (
                    "[Mature] "
                    if node.skeleton_str in self._current_mature_formulas
                    else ""
                )
                self._log(
                    f"\n  [{pi+1}/{len(parent_ids)}] {mature_tag}"
                    f"Parent: {parent_id}\n"
                    f"    train={self._fmt(node.train_nmse)}"
                    f"  test={self._fmt(node.test_nmse)}"
                    f"  ood={self._fmt(node.ood_test_nmse)}"
                )
                candidates = self._generate_candidates(parent_id, node)
                if self.shutdown_requested():
                    return None
                if not candidates:
                    self._log("    No valid candidate mutations, skipping.")
                    self._final_sweep_add_ids(
                        "skipped_parent_ids", [parent_id])
                    self._final_sweep_add_ids(
                        "completed_parent_ids", [parent_id])
                    self.save_checkpoint(
                        stage="final_sweep_parent_skipped",
                        next_step=self._resume_step,
                        force=True,
                    )
                    return None
                if self.shutdown_requested():
                    return None
                return (parent_id, node, candidates)

            try:
                def _submit_sweep_eval(ret):
                    if self.shutdown_requested():
                        return False
                    parent_id = ret[0]
                    self._final_sweep_add_ids(
                        "generated_parent_ids", [parent_id])
                    self.save_checkpoint(
                        stage="final_sweep_parent_generated",
                        next_step=self._resume_step,
                        force=True,
                    )
                    eval_futs[eval_pool.submit(
                        self._evaluate_all_parents,
                        [ret],
                        "final_sweep",
                        [parent_id],
                    )] = parent_id
                    return True

                if n_sweep_workers <= 1:
                    for node in remaining_nodes:
                        if self.shutdown_requested():
                            break
                        ret = _sweep_one(node)
                        if ret is not None:
                            _submit_sweep_eval(ret)
                else:
                    tpool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=n_sweep_workers)
                    try:
                        futs = {
                            tpool.submit(_sweep_one, node): node.skeleton_str
                            for node in remaining_nodes
                        }
                        for fut in self._completed_futures(futs):
                            ret = fut.result()
                            if ret is not None:
                                _submit_sweep_eval(ret)
                    finally:
                        tpool.shutdown(
                            wait=not self.shutdown_requested(),
                            cancel_futures=self.shutdown_requested(),
                        )

                for fut in self._completed_futures(
                    eval_futs,
                    wait_label="final_sweep_parent_eval",
                    future_owners=eval_futs,
                ):
                    parent_id = eval_futs[fut]
                    sweep_results.extend(fut.result())
                    self._final_sweep_add_ids(
                        "completed_parent_ids", [parent_id])
                    self.save_checkpoint(
                        stage="final_sweep_parent_complete",
                        next_step=self._resume_step,
                        force=True,
                    )
            finally:
                eval_pool.shutdown(
                    wait=not self.shutdown_requested(),
                    cancel_futures=self.shutdown_requested(),
                )

        if self.shutdown_requested():
            self.save_checkpoint(
                status="interrupted",
                stage="interrupted_during_final_sweep",
                next_step=self._resume_step,
                force=True,
            )
            return

        if sweep_results:
            self._mark_mature_nodes()
            n_valid = sum(
                1 for _, _, cn, _ in sweep_results
                if not cn.is_degenerated
            )
            self._log(
                f"\n  [Final sweep complete] Evaluated {len(sweep_results)}, "
                f"{n_valid} valid"
            )

        if self._final_sweep_state:
            self._final_sweep_state["active"] = False
        self._active_step = None
        self.save_checkpoint(
            stage="final_sweep_complete",
            next_step=self._resume_step,
            force=True,
        )

    # ------------------------------------------------------------------ #
    # Main evolution loop
    # ------------------------------------------------------------------ #

    def run(self) -> str:
        """Execute the main evolution tree search loop."""
        if not self.tree.all_nodes:
            raise RuntimeError("Evolution tree is empty; call initialize_seeds() first.")

        self._log("=" * 60)
        self._log("Evolution tree search started")
        self._log(
            f"[Early stop threshold] train<{self.mature_train_threshold:.2e}"
        )
        self._log("=" * 60)
        start_time = time.time()

        if self._resume_step > 0:
            self._log(
                f"[checkpoint] Continuing from step {self._resume_step + 1}/{self.max_steps}"
            )

        mature = self._mark_mature_nodes(log_new=False)
        if (
            self.refine_output
            and self._resume_step > 0
            and len(mature) >= self.max_mature_nodes
        ):
            self._log(
                f"[checkpoint] Resume state already has {len(mature)} mature nodes"
                f" (>= {self.max_mature_nodes}); skipping evolution loop "
                "and entering final sweep."
            )
            for i, m in enumerate(sorted(mature, key=lambda n: n.train_nmse)):
                self._log(
                    f"    {i+1}. {m.skeleton_str}\n"
                    f"       train={self._fmt(m.train_nmse)}  "
                    f"test={self._fmt(m.test_nmse)}  "
                    f"ood={self._fmt(m.ood_test_nmse)}"
                )
        else:
            for step in range(self._resume_step, self.max_steps):
                if self.shutdown_requested():
                    self._log("[checkpoint] Shutdown requested before next step; saving interrupted state.")
                    self.save_checkpoint(
                        status="interrupted",
                        stage="interrupted_between_steps",
                        next_step=step,
                    )
                    break

                self._resume_step = step
                self.save_checkpoint(stage="step_started", next_step=step)
                self._log(f"\n{'='*60}")
                self._log(f"Step {step + 1}/{self.max_steps}")
                # Nodes created from here on are stamped with this round number
                # (used to report when each Final-output formula first appeared).
                self.tree.current_step = step + 1
                self._log("=" * 60)

                all_step_results: List[
                    Tuple[str, EvalResult, EvolutionNode, str]
                ] = []
                eval_futs: Dict[
                    concurrent.futures.Future,
                    str,
                ] = {}

                if self._active_step_matches(step):
                    parent_ids_used = list(self._active_step.get("parent_ids", []))
                    all_step_results = self._results_from_active_step()
                    parent_cands = self._parent_cands_from_active_step()
                    self._log(
                        f"[checkpoint] Resuming step {step + 1}: "
                        f"{len(all_step_results)} evaluated, "
                        f"{sum(len(cands) for _, _, cands in parent_cands)} pending"
                    )
                    if parent_cands:
                        new_results = self._evaluate_all_parents(
                            parent_cands,
                            phase="step_eval",
                            parent_ids=parent_ids_used,
                        )
                        if self._active_step_matches(step):
                            all_step_results = self._results_from_active_step()
                        else:
                            all_step_results.extend(new_results)
                else:
                    parent_ids_used = []
                    self._active_step = None

                    exhausted_ids = {
                        fid for fid, cnt in self._mature_select_count.items()
                        if cnt >= self.mature_anneal_budget
                    }
                    mature = self._mark_mature_nodes(log_new=False)
                    exhausted_ids.update({
                        node.skeleton_str for node in mature
                        if self._mature_select_count.get(
                            node.skeleton_str, 0) >= self.mature_anneal_budget
                    })
                    summary = self.tree.get_tree_summary(
                        max_nodes=self.selector_context_size,
                        exclude_formulas=exhausted_ids,
                    )
                    if len(summary) < 1:
                        self._log("  All available nodes have exhausted their budget, skipping.")
                        self._resume_step = step + 1
                        self.save_checkpoint(stage="step_complete", next_step=step + 1)
                        continue

                    plans = self.selector.plan(
                        tree_summary=summary,
                        context_prompt=self.dataset.get_context_prompt(),
                        candidate_num=self.candidate_num,
                        selection_history=self._selection_history,
                    )
                    self.save_checkpoint(stage="selector_planned", next_step=step)

                    self._log(f"\n  [Selector] Selected {len(plans)} parents:")

                    n_workers = min(self.n_parent_workers, len(plans))
                    eval_pool = concurrent.futures.ThreadPoolExecutor(
                        max_workers=max(1, n_workers))
                    try:
                        parent_ids_seen: Set[str] = set()

                        def _submit_parent_eval(ret):
                            if self.shutdown_requested():
                                return False
                            if not ret[2]:
                                return False
                            if ret[0] not in parent_ids_seen:
                                parent_ids_seen.add(ret[0])
                                parent_ids_used.append(ret[0])
                            eval_futs[eval_pool.submit(
                                self._evaluate_all_parents,
                                [ret],
                                "step_eval",
                                [ret[0]],
                            )] = ret[0]
                            return True

                        def _submit_programmatic_and_llm(
                            ret,
                            tpool: concurrent.futures.ThreadPoolExecutor,
                            llm_futs: Dict[concurrent.futures.Future, str],
                        ) -> None:
                            if ret is None or self.shutdown_requested():
                                return
                            parent_id, parent_node, candidates, auto_descs, seen = ret
                            if candidates:
                                _submit_parent_eval(
                                    (parent_id, parent_node, candidates))
                            llm_futs[tpool.submit(
                                self._generate_llm_batch_for_parent,
                                parent_id,
                                parent_node,
                                auto_descs,
                                seen,
                            )] = parent_id

                        tpool = concurrent.futures.ThreadPoolExecutor(
                            max_workers=max(1, n_workers))
                        try:
                            programmatic_futs: Dict[
                                concurrent.futures.Future, int
                            ] = {
                                tpool.submit(
                                    self._generate_programmatic_for_parent,
                                    pi, len(plans), plan,
                                ): pi
                                for pi, plan in enumerate(plans)
                            }
                            llm_futs: Dict[concurrent.futures.Future, str] = {}
                            _step_wait_diag_t = time.monotonic()
                            while (
                                programmatic_futs
                                or llm_futs
                            ):
                                if self.shutdown_requested():
                                    for fut in programmatic_futs:
                                        fut.cancel()
                                    for fut in llm_futs:
                                        fut.cancel()
                                    break
                                wait_set = (
                                    set(programmatic_futs)
                                    | set(llm_futs)
                                )
                                done, _ = concurrent.futures.wait(
                                    wait_set,
                                    timeout=0.5,
                                    return_when=concurrent.futures.FIRST_COMPLETED,
                                )
                                if not done:
                                    _now = time.monotonic()
                                    if _now - _step_wait_diag_t >= 30.0:
                                        _step_wait_diag_t = _now
                                        self._log(
                                            f"  [StepWaitDiag] step={step + 1}"
                                            f" programmatic={len(programmatic_futs)}"
                                            f" llm={len(llm_futs)}"
                                            f" eval={len(eval_futs)}"
                                        )
                                        self.save_checkpoint(
                                            stage="step_llm_wait",
                                            next_step=step,
                                        )
                                    continue
                                for fut in done:
                                    if fut in programmatic_futs:
                                        programmatic_futs.pop(fut, None)
                                        try:
                                            ret = fut.result()
                                        except Exception as _prog_exc:
                                            self._log(
                                                f"  [Programmatic] Exception in candidate generation: {_prog_exc}"
                                            )
                                            ret = None
                                        _submit_programmatic_and_llm(
                                            ret, tpool, llm_futs)
                                        continue
                                    llm_futs.pop(fut, None)
                                    try:
                                        ret = fut.result()
                                    except Exception as _llm_exc:
                                        self._log(
                                            f"  [LLM] Exception in batch generation: {_llm_exc}"
                                        )
                                        ret = None
                                    if ret is not None:
                                        _submit_parent_eval(ret)
                        finally:
                            tpool.shutdown(
                                wait=not self.shutdown_requested(),
                                cancel_futures=self.shutdown_requested(),
                            )

                        for fut in self._completed_futures(
                            eval_futs,
                            wait_label="step_parent_eval",
                            future_owners=eval_futs,
                        ):
                            all_step_results.extend(fut.result())
                    finally:
                        eval_pool.shutdown(
                            wait=not self.shutdown_requested(),
                            cancel_futures=self.shutdown_requested(),
                        )

                if self.shutdown_requested():
                    self._log("[checkpoint] Shutdown requested after evaluation; saving interrupted state.")
                    self.save_checkpoint(
                        status="interrupted",
                        stage="interrupted_after_eval",
                        next_step=step,
                    )
                    break

                self._selection_history.append(parent_ids_used)

                improved_children = []
                for formula, result, child_node, mutation in all_step_results:
                    if child_node.is_degenerated:
                        continue
                    parent_node = self.tree.get_node(child_node.parent_id)
                    parent_train = parent_node.train_nmse if parent_node else float("inf")
                    if result.train_nmse < parent_train:
                        improved_children.append((child_node, result))

                if self.enable_describe and improved_children:
                    desc_items = [
                        {
                            "formula": child_node.skeleton_str,
                            "fitted_params_str": child_node.get_fitted_params_str(),
                            "mse": result.train_nmse,
                        }
                        for child_node, result in improved_children
                    ]
                    descriptions = self.generator.describe_batch(desc_items)
                    for (child_node, _), desc in zip(improved_children, descriptions):
                        if desc:
                            child_node.description = desc

                mature = self._log_step_summary(
                    step, all_step_results, parent_ids_used)
                self._resume_step = step + 1
                self._active_step = None
                self.save_checkpoint(stage="step_complete", next_step=step + 1)

                if len(mature) >= self.max_mature_nodes:
                    self._log(
                        f"\n  [Early stop] Collected {len(mature)} mature nodes"
                        f" (>= {self.max_mature_nodes}), ending search early."
                    )
                    for i, m in enumerate(sorted(mature, key=lambda n: n.train_nmse)):
                        self._log(
                            f"    {i+1}. {m.skeleton_str}\n"
                            f"       train={self._fmt(m.train_nmse)}  "
                            f"test={self._fmt(m.test_nmse)}  "
                            f"ood={self._fmt(m.ood_test_nmse)}"
                        )
                    break

        # ------------------------------------------------------------ #
        # Final sweep: ensure all mature + supplement nodes were selected
        # as parent at least once (only when refine_output=True)
        # ------------------------------------------------------------ #
        self._run_final_sweep()

        # ------------------------------------------------------------ #
        # Final report: all mature nodes + supplement nodes
        # ------------------------------------------------------------ #
        elapsed = time.time() - start_time
        stats = self.tree.get_stats()
        best = self.tree.best_node
        best_params = best.get_fitted_params_str() if best else ""

        self._log("\n" + "=" * 60)
        done_label = "completed" if not self.shutdown_requested() else "interrupted"
        self._log(f"Search {done_label}, elapsed {elapsed:.1f}s")
        self._log(
            f"Explored {stats['total_nodes']} nodes total, "
            f"global best (train): train={self._fmt(stats['best_train_nmse'])}"
            f"  test={self._fmt(stats['best_test_nmse'])}"
            f"  ood={self._fmt(stats['best_ood_test_nmse'])}"
        )
        self._log(f"Best formula: {stats['best_expr']}")
        self._log(f"Best expression params: {best_params or 'no constant params'}")

        final_mature = self._mark_mature_nodes(log_new=False)
        final_supplement: List[EvolutionNode] = []
        if len(final_mature) < self.max_mature_nodes:
            final_supplement = sorted(
                [n for n in self.tree.all_nodes
                 if n.is_evaluated
                 and n.skeleton_str not in self._current_mature_formulas
                 and not n.is_degenerated],
                key=lambda n: n.train_nmse,
            )[:self.max_mature_nodes - len(final_mature)]

        final_top = final_mature + final_supplement
        self._log(
            f"\nFinal output {len(final_top)} formulas"
            f" (mature {len(final_mature)}, supplement {len(final_supplement)}):"
        )
        for i, n in enumerate(final_top):
            mature_tag = (
                " [Mature]"
                if n.skeleton_str in self._current_mature_formulas
                else ""
            )
            # Round at which this structure first appeared (0/None = seed, before Step 1).
            cs = getattr(n, "created_step", None)
            round_tag = "seed" if not cs else str(cs)
            self._log(
                f"  {i+1}.{mature_tag} <<<FORMULA>>>{n.skeleton_str}<<<END_FORMULA>>>"
                f"  [first appeared: round {round_tag}]\n"
                f"     params: {n.get_fitted_params_str() or 'none'}\n"
                f"     train={self._fmt(n.train_nmse)}  "
                f"test={self._fmt(n.test_nmse)}  "
                f"ood={self._fmt(n.ood_test_nmse)}"
            )

        self._log("=" * 60)
        if not self.shutdown_requested():
            self.save_checkpoint(
                status="completed",
                stage="completed",
                next_step=min(self._resume_step, self.max_steps),
            )

        return stats["best_expr"] or ""

    # ------------------------------------------------------------------ #
    # Phase 1: Validate parent + generate candidates
    # ------------------------------------------------------------------ #

    def _resolve_parent_for_generation(
        self,
        pi: int,
        total: int,
        plan: Dict[str, Any],
    ) -> Optional[Tuple[str, EvolutionNode]]:
        """Validate and log the selected parent for candidate generation."""
        external_parent_id = plan.get("parent_id")
        selector_id = plan.get("selector_id", external_parent_id)
        parent_id = str(plan.get("parent_formula", "") or "").strip()
        rationale = plan.get("rationale", "")

        parent_node = self.tree.get_node(parent_id) if parent_id else None
        if parent_node is None:
            parent_node = self.tree.get_node_by_id(external_parent_id)
            if parent_node is not None:
                parent_id = parent_node.skeleton_str
        if parent_node is None and isinstance(external_parent_id, str):
            parent_node = self.tree.get_node(external_parent_id)
            if parent_node is not None:
                parent_id = parent_node.skeleton_str
        if parent_node is None:
            best = self.tree.best_node
            if best is None:
                return None
            parent_node = best
            parent_id = best.skeleton_str
            selector_id = parent_node.node_id

        if parent_id in self._current_mature_formulas:
            used = self._mature_select_count.get(parent_id, 0)
            if used >= self.mature_anneal_budget:
                self._log(
                    f"\n  [{pi+1}/{total}] Parent: {parent_id}\n"
                    f"    Mature and annealing budget exhausted ({used}/{self.mature_anneal_budget}), skipping."
                )
                return None
            self._mature_select_count[parent_id] = used + 1
            self._log(
                f"\n  [{pi+1}/{total}] Parent: {parent_id}\n"
                f"    [Annealing] Mature node, remaining budget "
                f"{self.mature_anneal_budget - used - 1}/{self.mature_anneal_budget}"
            )

        if parent_node.is_degenerated:
            self._log(
                f"\n  [{pi+1}/{total}] Parent: {parent_id}\n"
                f"    Degenerated, skipping."
            )
            return None

        parent_params_str = parent_node.get_fitted_params_str()
        display_id = selector_id if selector_id is not None else parent_node.node_id
        selector_part = f" [id={display_id}]" if display_id is not None else ""
        self._log(
            f"\n  [{pi+1}/{total}] Parent{selector_part}: {parent_id}\n"
            f"    params: {parent_params_str or 'no constant params'}\n"
            f"    train={self._fmt(parent_node.train_nmse)}"
            f"  test={self._fmt(parent_node.test_nmse)}"
            f"  ood={self._fmt(parent_node.ood_test_nmse)}\n"
            f"    rationale: {rationale}"
        )

        return parent_id, parent_node

    @staticmethod
    def _dedup_raw_candidates(
        candidates: List[Dict[str, Any]],
        seen: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        seen_keys = set(seen or set())
        unique: List[Dict[str, Any]] = []
        for cand in candidates:
            key = str(cand.get("expression", "")).strip()
            if key and key not in seen_keys:
                seen_keys.add(key)
                unique.append(cand)
        return unique, seen_keys

    def _generate_programmatic_candidates(
        self,
        parent_id: str,
    ) -> Tuple[List[Dict[str, Any]], List[str], Set[str]]:
        if self.skip_programmatic_mutations:
            deletion_cands: List[Dict[str, Any]] = []
            addition_cands: List[Dict[str, Any]] = []
        else:
            deletion_cands = self.mutator.enumerate_deletions(parent_id)
            addition_cands = self.mutator.enumerate_additions(parent_id)

        self._log(
            f"\n  [Mutation] Programmatic deletion: {len(deletion_cands)}, "
            f"addition: {len(addition_cands)}"
        )

        auto_cands = deletion_cands + addition_cands
        auto_mutation_descs = [c["mutation"] for c in auto_cands]
        unique, seen = self._dedup_raw_candidates(auto_cands)
        self._log(
            f"  [Mutation] Programmatic ready: {len(unique)} unique candidates"
            + (f" (raw {len(auto_cands)})" if len(unique) < len(auto_cands) else "")
        )
        return unique, auto_mutation_descs, seen

    def _generate_llm_candidates(
        self,
        parent_id: str,
        parent_node: EvolutionNode,
        auto_mutation_descs: List[str],
        seen: Optional[Set[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], Set[str]]:
        if getattr(self.llm_mutator, "strip_ast_prompt", False):
            labeled_ast = ""
        else:
            labeled_ast = self.mutator.get_labeled_ast(parent_id)

        with self._lock:
            top_exprs = self._get_top_exprs(k=self.mutator_seen_topk)

        llm_cands = self.llm_mutator.suggest_mutations(
            context_prompt=self.dataset.get_context_prompt(),
            parent_formula=parent_id,
            labeled_ast=labeled_ast,
            parent_fitted_params=parent_node.get_fitted_params_str(),
            auto_mutations=auto_mutation_descs,
            variables=self.dataset.feature_names,
            top_exprs=top_exprs,
        )

        for c in llm_cands:
            c["mutation"] = f"[LLM] {c.get('mutation', 'custom mutation')}"
            expr = self._freshen_llm_added_term_params(
                parent_id,
                c.get("expression", ""),
            )
            c["expression"] = expr

        self._log(f"  [Mutation] LLM suggestions: {len(llm_cands)}")
        unique, seen_after = self._dedup_raw_candidates(llm_cands, seen)
        self._log(
            f"  [Mutation] LLM ready: {len(unique)} unique candidates"
            + (f" (raw {len(llm_cands)})" if len(unique) < len(llm_cands) else "")
        )
        return unique, seen_after

    def _generate_programmatic_for_parent(
        self,
        pi: int,
        total: int,
        plan: Dict[str, Any],
    ) -> Optional[
        Tuple[str, EvolutionNode, List[Dict[str, Any]], List[str], Set[str]]
    ]:
        resolved = self._resolve_parent_for_generation(pi, total, plan)
        if resolved is None:
            return None
        parent_id, parent_node = resolved
        candidates, auto_descs, seen = self._generate_programmatic_candidates(
            parent_id)
        if not candidates:
            self._log("    No valid programmatic candidate mutations.")
        return parent_id, parent_node, candidates, auto_descs, seen

    def _generate_llm_batch_for_parent(
        self,
        parent_id: str,
        parent_node: EvolutionNode,
        auto_mutation_descs: List[str],
        seen: Optional[Set[str]] = None,
    ) -> Optional[Tuple[str, EvolutionNode, List[Dict[str, Any]]]]:
        candidates, _ = self._generate_llm_candidates(
            parent_id,
            parent_node,
            auto_mutation_descs,
            seen,
        )
        if not candidates:
            self._log("    No valid LLM candidate mutations.")
            return None
        return parent_id, parent_node, candidates

    def _generate_for_parent(
        self,
        pi: int,
        total: int,
        plan: Dict[str, Any],
    ) -> Optional[Tuple[str, EvolutionNode, List[Dict[str, Any]]]]:
        """Validate parent node and generate all candidate mutations."""
        resolved = self._resolve_parent_for_generation(pi, total, plan)
        if resolved is None:
            return None
        parent_id, parent_node = resolved

        candidates = self._generate_candidates(parent_id, parent_node)
        if not candidates:
            self._log("    No valid candidate mutations, skipping.")
            return None
        return (parent_id, parent_node, candidates)

    # ------------------------------------------------------------------ #
    # Phase 2: Unified evaluation
    # ------------------------------------------------------------------ #

    def _evaluate_all_parents(
        self,
        parent_cands: List[Tuple[str, EvolutionNode, List[Dict[str, Any]]]],
        phase: str = "eval",
        parent_ids: Optional[List[str]] = None,
    ) -> List[Tuple[str, EvalResult, EvolutionNode, str]]:
        """Pool candidates from multiple parents and evaluate them with a shared worker pool."""

        PendingItem = Tuple[str, List[str], str, EvolutionNode, EvolutionNode, str]
        RawPendingItem = Tuple[
            str, List[str], str, str, EvolutionNode, str, bool
        ]
        raw_pending: List[RawPendingItem] = []

        for pi, (parent_id, parent_node, candidates) in enumerate(parent_cands):
            tag = f"P{pi+1}"
            for cand in candidates:
                formula = str(cand.get("expression", "")).strip()
                if not formula:
                    continue
                raw_pending.append(
                    (
                        formula,
                        list(cand.get("params", [])),
                        str(cand.get("mutation", "")),
                        parent_id,
                        parent_node,
                        tag,
                        bool(cand.get("_resume_active")),
                    )
                )

        if not raw_pending:
            return []

        results: List[Tuple[str, EvalResult, EvolutionNode, str]] = []
        if self.eval_executor is None:
            pool_cm = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.n_eval_workers)
            pool = pool_cm.__enter__()
            close_pool = True
        else:
            pool = self.eval_executor
            pool_cm = None
            close_pool = False

        skipped_count = 0
        prepared_count = 0
        submitted_total = 0
        initial_submitted = 0
        normalize_timeout = self.normalize_timeout
        max_normalize_futures = 8
        try:
            normalize_futures: Dict[Any, RawPendingItem] = {}
            eval_futures: Dict[Any, PendingItem] = {}
            degen_futures: Dict[Any, Tuple[
                PendingItem,
                str,
                EvalResult,
            ]] = {}
            raw_index = 0
            seen_batch_keys: Set[str] = set()

            tag_map: Dict[str, str] = {}
            for _, _, _, _, pnode, tag, _ in raw_pending:
                if tag not in tag_map:
                    tag_map[tag] = pnode.skeleton_str
            tag_legend = "  ".join(
                f"{t}={p}" for t, p in sorted(tag_map.items()))
            self._debug_log(
                f"\n  [Unified eval] streaming up to {len(raw_pending)} raw candidates, "
                f"{self._eval_pool_label()}, timeout={self.timeout:.0f}s, "
                f"normalize_timeout={normalize_timeout:.0f}s, "
                f"max_normalize_futures={max_normalize_futures}\n"
                f"    Parent tags: {tag_legend}"
            )

            def _submit_item(
                item: PendingItem,
                sp_expr: sp.Expr,
            ) -> bool:
                if self.shutdown_requested():
                    return False
                formula, params, mutation, child_node, parent_node, tag = item
                parent_params = (
                    parent_node.fitted_params
                    if parent_node.fitted_params else None
                )
                self._candidate_statuses[formula] = "queued"
                try:
                    fut = self._submit_candidate_eval(
                        pool,
                        self.evaluator,
                        sp_expr,
                        formula,
                        params,
                        parent_params,
                        self.timeout,
                    )
                except RuntimeError:
                    self._candidate_statuses[formula] = "pending"
                    if self.shutdown_requested():
                        return False
                    raise
                eval_futures[fut] = item
                return True

            def _submit_degen_for_item(
                item: PendingItem,
                result_formula: str,
                result_param_names: List[str],
                result_key: str,
                result: EvalResult,
                norm_params: List[float],
            ) -> None:
                fut = self._submit_degeneracy_check(
                    pool,
                    self.evaluator,
                    self.mutator,
                    result_formula,
                    result_param_names,
                    norm_params,
                    self.degeneracy_timeout,
                )
                degen_futures[fut] = (
                    item,
                    result_key,
                    result,
                )

            def _submit_more_normalize() -> None:
                nonlocal raw_index
                while (
                    raw_index < len(raw_pending)
                    and len(normalize_futures) < max_normalize_futures
                    and not self.shutdown_requested()
                ):
                    raw_item = raw_pending[raw_index]
                    raw_index += 1
                    raw_formula, raw_params, *_ = raw_item
                    try:
                        fut = self._submit_normalization(
                            pool,
                            self.dataset.feature_names,
                            raw_formula,
                            raw_params,
                            False,
                            normalize_timeout,
                        )
                    except RuntimeError:
                        if self.shutdown_requested():
                            break
                        raise
                    normalize_futures[fut] = raw_item

            def _claim_and_submit_normalized(future: Any) -> None:
                nonlocal skipped_count, prepared_count
                nonlocal submitted_total, initial_submitted
                raw_item = normalize_futures.pop(future)
                (
                    raw_formula,
                    raw_params,
                    mutation,
                    parent_id,
                    parent_node,
                    tag,
                    allow_active_reclaim,
                ) = raw_item
                try:
                    norm_result = future.result()
                except Exception as e:
                    self._log(
                        f"    ✗ [{tag}|{mutation}]: "
                        f"normalization exception ({e}) -> "
                        f"{raw_formula}")
                    return
                formula, sp_expr_prepared, params, canonical_key, norm_status = norm_result
                if norm_status not in {"ok", "timeout_raw", "timeout_softkey"}:
                    self._debug_log(
                        f"    ✗ [{tag}|{mutation}]: "
                        f"normalization {norm_status} -> "
                        f"{raw_formula}")
                    return
                if norm_status == "timeout_softkey":
                    self._log(
                        f"    ⚠ [{tag}|{mutation}]: "
                        f"normalization timeout; evaluating raw expression "
                        f"(dedup key kept) -> {raw_formula}")
                elif norm_status == "timeout_raw":
                    self._log(
                        f"    ⚠ [{tag}|{mutation}]: "
                        f"normalization timeout; evaluating raw expression -> "
                        f"{raw_formula}")
                if not formula or sp_expr_prepared is None:
                    return
                prepared_count += 1
                key_for_batch = canonical_key or formula
                if key_for_batch in seen_batch_keys:
                    return
                seen_batch_keys.add(key_for_batch)

                with self._lock:
                    child_node, params, skipped_params = (
                        self._claim_candidate_node(
                            formula,
                            params,
                            parent_formula=parent_id,
                            allow_active_reclaim=allow_active_reclaim,
                            canonical_key=canonical_key,
                        )
                    )
                if skipped_params:
                    skipped_count = skipped_count + 1
                    return
                if child_node is None:
                    return
                item: PendingItem = (
                    formula, params, mutation, child_node, parent_node, tag)
                with self._lock:
                    self._record_active_candidates(
                        step=self._resume_step,
                        phase=phase,
                        parent_ids=[parent_id],
                        pending=[item],
                    )
                self.save_checkpoint(
                    stage=f"{phase}_candidates_pending",
                    next_step=self._resume_step,
                    force=(phase == "final_sweep"),
                )
                if _submit_item(item, sp_expr_prepared):
                    submitted_total += 1
                    initial_submitted += 1

            def _enqueue_simplified(
                cand: Dict[str, Any],
                parent_node: EvolutionNode,
                tag: str,
                sp_expr: Optional[sp.Expr] = None,
            ) -> bool:
                formula = str(cand.get("expression", "")).strip()
                params = list(cand.get("params", []))
                mutation = str(cand.get("mutation", ""))
                canonical_key = str(cand.get("canonical_key", ""))
                if not formula or len(params) > self.max_params:
                    return False

                # O(1) fast-path: skip lock acquisition if already seen
                key_for_check = canonical_key or formula
                if key_for_check in self._seen_keys:
                    return False

                resolved_sp_expr = sp_expr if sp_expr is not None else self.normalizer.parse(formula)
                if resolved_sp_expr is None:
                    return False

                with self._lock:
                    child_node, params, _ = self._claim_candidate_node(
                        formula,
                        params,
                        parent_formula=parent_node.skeleton_str,
                        canonical_key=canonical_key,
                    )
                    if child_node is None:
                        existing = self._find_existing_equivalent_node(
                            formula, canonical_key=canonical_key)
                        if existing is not None:
                            self.tree.attach_child(
                                parent_node.skeleton_str,
                                existing.skeleton_str,
                            )
                            if self._child_hides_parent(
                                parent_node, existing):
                                parent_node.mark_degenerated(
                                    existing.skeleton_str,
                                    ["hidden by existing degenerated child"],
                                    kind="simplified",
                                    child_train_nmse=existing.train_nmse,
                                    child_test_nmse=existing.test_nmse,
                                    child_ood_test_nmse=existing.ood_test_nmse,
                                )
                                self._log_unlocked(
                                    f"    ↳ parent marked degenerated by "
                                    f"existing simplified child: "
                                    f"{parent_node.skeleton_str} -> "
                                    f"{existing.skeleton_str}"
                                )
                        return False
                    item: PendingItem = (
                        formula,
                        params,
                        mutation,
                        child_node,
                        parent_node,
                        tag,
                    )
                    self._record_active_candidates(
                        step=self._resume_step,
                        phase=phase,
                        parent_ids=[parent_node.skeleton_str],
                        pending=[item],
                    )
                return _submit_item(item, resolved_sp_expr)

            def _maybe_log_eval_diagnostics(force: bool = False) -> None:
                now = time.monotonic()
                if (
                    not force
                    and self._last_eval_diagnostic_monotonic > 0
                    and now - self._last_eval_diagnostic_monotonic < 30.0
                ):
                    return
                self._last_eval_diagnostic_monotonic = now
                self._diagnostic_log(
                    "    "
                    + self._eval_diagnostic_summary(
                        normalize_futures=len(normalize_futures),
                        eval_futures=len(eval_futures),
                        degen_futures=len(degen_futures),
                        raw_remaining=max(0, len(raw_pending) - raw_index),
                        submitted_total=submitted_total,
                        prepared_count=prepared_count,
                        skipped_count=skipped_count,
                    )
                )

            _submit_more_normalize()
            _maybe_log_eval_diagnostics(force=True)
            while (
                normalize_futures
                or eval_futures
                or degen_futures
                or raw_index < len(raw_pending)
            ):
                if self.shutdown_requested():
                    for fut in normalize_futures:
                        fut.cancel()
                    for fut in eval_futures:
                        fut.cancel()
                    for fut in degen_futures:
                        fut.cancel()
                    break
                _submit_more_normalize()
                wait_set = (
                    set(normalize_futures)
                    | set(eval_futures)
                    | set(degen_futures)
                )
                if not wait_set:
                    break
                done, _ = concurrent.futures.wait(
                    wait_set,
                    timeout=0.5,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                if not done:
                    self.save_checkpoint(
                        stage=f"{phase}_eval_wait",
                        next_step=self._resume_step,
                    )
                    _maybe_log_eval_diagnostics()
                    continue
                for future in done:
                    if future in normalize_futures:
                        _claim_and_submit_normalized(future)
                        continue

                    if future in eval_futures:
                        item = eval_futures.pop(future)
                        formula, params, mutation, child_node, parent_node, tag = item
                        self._candidate_statuses[formula] = "running"
                        try:
                            (
                                result_formula,
                                result_param_names,
                                result_key,
                                result,
                                norm_params,
                            ) = future.result()
                        except Exception as e:
                            self._candidate_statuses[formula] = "exception"
                            self._log(
                                f"    ✗ [{tag}|{mutation}]: "
                                f"exception ({e}) -> {formula}")
                            self.save_checkpoint(
                                stage="candidate_exception",
                                next_step=self._resume_step,
                            )
                            continue

                        if result.train_nmse == float("inf"):
                            self._candidate_statuses[formula] = "failed"
                            reason = (
                                getattr(result, "fail_reason", "") or "unknown"
                            )
                            self._debug_log(
                                f"    ✗ [{tag}|{mutation}]: "
                                f"eval rejected ({reason}) -> {formula}")
                            self.save_checkpoint(
                                stage="candidate_failed",
                                next_step=self._resume_step,
                            )
                            continue

                        with self._lock:
                            self.tree.update_score(
                                result_formula,
                                train_nmse=result.train_nmse,
                                test_nmse=result.test_nmse,
                                ood_test_nmse=result.ood_test_nmse,
                                param_names=result_param_names,
                                fitted_params=norm_params,
                            )
                            if "[Degenerated]" in mutation:
                                dominates = (
                                    result.train_nmse
                                    <= parent_node.train_nmse
                                    * DegenerationConfig().dominance_ratio
                                )
                                child_mature = (
                                    result.train_nmse
                                    < self.mature_train_threshold
                                )
                                if dominates or child_mature:
                                    parent_node.mark_degenerated(
                                        formula,
                                        [
                                            f"dominated by degenerated child train={self._fmt(result.train_nmse)}"
                                        ],
                                        kind="simplified",
                                        child_train_nmse=result.train_nmse,
                                        child_test_nmse=result.test_nmse,
                                        child_ood_test_nmse=result.ood_test_nmse,
                                    )
                                    self._log_unlocked(
                                        f"    ↳ parent marked degenerated: "
                                        f"{parent_node.skeleton_str} -> {formula}"
                                    )
                        _submit_degen_for_item(
                            item,
                            result_formula,
                            result_param_names,
                            result_key,
                            result,
                            norm_params,
                        )
                        continue

                    item, result_key, result = degen_futures.pop(future)
                    formula, params, mutation, child_node, parent_node, tag = item
                    simplified_to_submit: Optional[Dict[str, Any]] = None
                    simp_sp_expr: Optional[sp.Expr] = None
                    try:
                        (
                            degen_status,
                            simp_expr,
                            simp_params,
                            simp_key,
                            degen_reasons,
                            simp_sp_expr,
                        ) = future.result()
                    except Exception as e:
                        degen_status, simp_expr, simp_params, simp_key = (
                            "ok", None, None, "")
                        degen_reasons = [f"degen_exception:{e}"]

                    with self._lock:
                        if degen_status == "timeout":
                            self._log_unlocked(
                                f"    ⚠ [{tag}|{mutation}]: "
                                f"degeneracy check timeout (>{self.degeneracy_timeout:.0f}s)"
                                f"  -> {formula}")
                            degen_status = "ok"

                        if degen_status == "interrupted":
                            self._candidate_statuses[formula] = "interrupted"
                            self.save_checkpoint(
                                stage="candidate_evaluated",
                                next_step=self._resume_step,
                            )
                            continue

                        if degen_status == "overfit":
                            child_node.mark_degenerated(
                                None,
                                degen_reasons,
                                kind="overfit",
                            )
                            self._candidate_statuses[formula] = "evaluated"
                            reason_str = (
                                '; '.join(degen_reasons)
                                if degen_reasons else "coeff>1e3"
                            )
                            self._log_unlocked(
                                f"    ✗ [{tag}|{mutation}]: "
                                f"overfitting ({reason_str})"
                                f"  -> {formula}"
                            )
                            self.save_checkpoint(
                                stage="candidate_evaluated",
                                next_step=self._resume_step,
                            )
                            continue

                        improved = result.train_nmse < parent_node.train_nmse
                        marker = "★" if improved else " "
                        self._log_unlocked(
                            f"  {marker} [{tag}|{mutation}]: "
                            f"test={self._fmt(result.test_nmse)}  "
                            f"train={self._fmt(result.train_nmse)}  "
                            f"ood={self._fmt(result.ood_test_nmse)}"
                            f"  → {formula}"
                        )

                        if degen_status == "simplified":
                            if simp_expr:
                                child_node.canonical_key = simp_key
                                child_node.add_degeneration_candidate(
                                    simp_expr,
                                    degen_reasons,
                                    kind="simplified",
                                )
                            self._candidate_statuses[formula] = "evaluated"
                            reason_str = (
                                '; '.join(degen_reasons)
                                if degen_reasons else ""
                            )
                            self._log_unlocked(
                                f"    ↳ degenerate simplified ({reason_str}): "
                                f"{formula} -> {simp_expr}"
                            )
                            simplified_to_submit = {
                                "expression": simp_expr,
                                "params": simp_params or [],
                                "canonical_key": simp_key,
                                "mutation": (
                                    f"[{tag}|[Degenerated] {mutation}]"
                                ),
                            }
                        else:
                            child_node.canonical_key = result_key
                            self._candidate_statuses[formula] = "evaluated"

                        self.save_checkpoint(
                            stage="candidate_evaluated",
                            next_step=self._resume_step,
                        )
                    results.append((formula, result, child_node, mutation))

                    if (
                        simplified_to_submit is not None
                        and not self.shutdown_requested()
                        and _enqueue_simplified(
                            simplified_to_submit, child_node, tag,
                            sp_expr=simp_sp_expr)
                    ):
                        submitted_total += 1

            if self.shutdown_requested() and (
                normalize_futures or eval_futures or degen_futures
            ):
                self.save_checkpoint(
                    status="interrupted",
                    stage=f"interrupted_during_{phase}_eval_wait",
                    next_step=self._resume_step,
                    force=(phase == "final_sweep"),
                )

            if skipped_count:
                self._log(
                    f"  [Candidate prepare] Prepared {prepared_count}, "
                    f"submitted {initial_submitted}, skipped {skipped_count} "
                    f"(params>{self.max_params})"
                )

            if submitted_total > initial_submitted:
                self._log(
                    f"  [Degenerate simplification] Submitted "
                    f"{submitted_total - initial_submitted} simplified formulas inline"
                )
        finally:
            if close_pool and pool_cm is not None:
                if self.shutdown_requested():
                    pool.shutdown(wait=False, cancel_futures=True)
                else:
                    pool_cm.__exit__(None, None, None)

        return results

    # ------------------------------------------------------------------ #
    # Preprocessing variant expansion
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Candidate generation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _param_sort_key(name: str) -> int:
        match = re.match(r"^c(\d+)$", name)
        return int(match.group(1)) if match else 10**9

    @staticmethod
    def _add_terms(expr: sp.Expr) -> List[sp.Expr]:
        return list(expr.args) if expr.func == sp.Add else [expr]

    def _terms_match(self, left: sp.Expr, right: sp.Expr) -> bool:
        if left == right:
            return True
        return self.normalizer.struct_hash(left) == self.normalizer.struct_hash(right)

    def _freshen_llm_added_term_params(
        self,
        parent_formula: str,
        candidate_formula: str,
    ) -> str:
        """Rename parent-parameter reuse only inside an LLM-added additive term.

        LLM mutation prompts require ``candidate = parent + new_term``.  When the
        LLM reuses parent symbols in ``new_term`` (for example, parent uses
        c0/c1/c2 and the new term is c0*cos(c1*log(A*c2 + 1))), those symbols
        accidentally tie the new term to the parent.  This alpha-renames only the
        added top-level terms while leaving the parent expression untouched.
        """
        try:
            parent_expr = sp.sympify(
                parent_formula,
                locals=self.normalizer.make_locals(parent_formula),
            )
            candidate_expr = sp.sympify(
                candidate_formula,
                locals=self.normalizer.make_locals(candidate_formula),
            )
        except Exception:
            return candidate_formula

        # sp.sympify may return a plain Python tuple (not a SymPy Expr) when the
        # input string contains commas, e.g. LLM output "P*c0*t, c1".  A Python
        # tuple has no .func attribute, so _add_terms would crash.  Guard here.
        if not isinstance(parent_expr, sp.Basic) or not isinstance(candidate_expr, sp.Basic):
            return candidate_formula

        parent_params = set(self.normalizer.collect_params(parent_expr))
        if not parent_params:
            return candidate_formula

        remaining_terms = self._add_terms(candidate_expr)
        for parent_term in self._add_terms(parent_expr):
            match_idx = None
            for i, cand_term in enumerate(remaining_terms):
                if self._terms_match(parent_term, cand_term):
                    match_idx = i
                    break
            if match_idx is None:
                # The candidate did not preserve parent as top-level additive
                # terms, so do not guess which symbols belong to the new term.
                return candidate_formula
            del remaining_terms[match_idx]

        if not remaining_terms:
            return candidate_formula

        added_expr = sp.Add(*remaining_terms)
        added_params = set(self.normalizer.collect_params(added_expr))
        conflicts = sorted(parent_params & added_params, key=self._param_sort_key)
        if not conflicts:
            return candidate_formula

        used_params = set(self.normalizer.collect_params(candidate_expr))
        next_idx = (
            max((self._param_sort_key(p) for p in used_params), default=-1) + 1
        )
        replacements: Dict[sp.Symbol, sp.Symbol] = {}
        for name in conflicts:
            while f"c{next_idx}" in used_params:
                next_idx += 1
            fresh_name = f"c{next_idx}"
            replacements[sp.Symbol(name)] = sp.Symbol(fresh_name)
            used_params.add(fresh_name)
            next_idx += 1

        fresh_added_expr = added_expr.xreplace(replacements)
        return str(parent_expr + fresh_added_expr)

    def _generate_candidates(
        self, parent_id: str, parent_node: EvolutionNode
    ) -> List[Dict[str, Any]]:
        """Generate all candidate mutations (programmatic deletion + programmatic addition + LLM suggestions)."""

        if self.skip_programmatic_mutations:
            deletion_cands: List[Dict[str, Any]] = []
            addition_cands: List[Dict[str, Any]] = []
        else:
            deletion_cands = self.mutator.enumerate_deletions(parent_id)
            addition_cands = self.mutator.enumerate_additions(parent_id)

        self._log(
            f"\n  [Mutation] Programmatic deletion: {len(deletion_cands)}, "
            f"addition: {len(addition_cands)}"
        )

        auto_mutation_descs = [
            c["mutation"]
            for c in deletion_cands + addition_cands
        ]
        if getattr(self.llm_mutator, "strip_ast_prompt", False):
            labeled_ast = ""
        else:
            labeled_ast = self.mutator.get_labeled_ast(parent_id)

        llm_cands = self.llm_mutator.suggest_mutations(
            context_prompt=self.dataset.get_context_prompt(),
            parent_formula=parent_id,
            labeled_ast=labeled_ast,
            parent_fitted_params=parent_node.get_fitted_params_str(),
            auto_mutations=auto_mutation_descs,
            variables=self.dataset.feature_names,
            top_exprs=self._get_top_exprs(k=self.mutator_seen_topk),
        )

        for c in llm_cands:
            c["mutation"] = f"[LLM] {c.get('mutation', 'custom mutation')}"
            expr = self._freshen_llm_added_term_params(
                parent_id,
                c.get("expression", ""),
            )
            c["expression"] = expr

        self._log(f"  [Mutation] LLM suggestions: {len(llm_cands)}")

        all_raw = deletion_cands + addition_cands + llm_cands
        seen_step: Set[str] = set()
        unique: List[Dict[str, Any]] = []
        for c in all_raw:
            key = str(c.get("expression", "")).strip()
            if key and key not in seen_step:
                seen_step.add(key)
                unique.append(c)

        self._log(
            f"  [Mutation] After dedup: {len(unique)} unique candidates"
            + (f" (raw {len(all_raw)})" if len(unique) < len(all_raw) else "")
        )
        return unique

    # ------------------------------------------------------------------ #
    # Logging and summary
    # ------------------------------------------------------------------ #

    def _log_step_summary(
        self,
        step: int,
        step_results: List[Tuple[str, EvalResult, EvolutionNode, str]],
        parent_ids: List[str],
    ) -> List[EvolutionNode]:
        self._log(
            f"\n  [Summary] Parents: {len(parent_ids)}, "
            f"successfully evaluated: {len(step_results)}"
        )

        best_of_step = None
        if step_results:
            best_formula, best_result, best_child, best_mutation = min(
                step_results, key=lambda x: x[1].train_nmse
            )
            best_of_step = best_formula
            self._log(
                f"  [Step best] {best_mutation}\n"
                f"    formula: {best_formula}\n"
                f"    params: {best_child.get_fitted_params_str() or 'none'}\n"
                f"    train={self._fmt(best_result.train_nmse)}"
                f"  test={self._fmt(best_result.test_nmse)}"
                f"  ood={self._fmt(best_result.ood_test_nmse)}"
            )

        best = self.tree.best_node
        if best:
            self._log(
                f"\n  [Global best] {best.skeleton_str}\n"
                f"    params: {best.get_fitted_params_str() or 'none'}\n"
                f"    train={self._fmt(best.train_nmse)}"
                f"  test={self._fmt(best.test_nmse)}"
                f"  ood={self._fmt(best.ood_test_nmse)}"
            )

        mature_nodes = self._mark_mature_nodes(log_new=True)
        n_mature = len(mature_nodes)
        if n_mature > 0:
            self._log(
                f"  [Mature nodes] {n_mature}/{self.max_mature_nodes}"
            )

        self.history.append({
            "step": step + 1,
            "parents": parent_ids,
            "n_evaluated": len(step_results),
            "best_of_step": best_of_step,
        })
        return mature_nodes

    # ------------------------------------------------------------------ #
    # Utility methods
    # ------------------------------------------------------------------ #

    def _is_node_mature(self, node: EvolutionNode) -> bool:
        return (node.is_evaluated
                and not node.is_degenerated
                and node.train_nmse < self.mature_train_threshold)

    def _child_hides_parent(
        self,
        parent: EvolutionNode,
        child: EvolutionNode,
    ) -> bool:
        if not child.is_evaluated:
            return False
        if child.is_degenerated:
            return False
        if child.train_nmse < self.mature_train_threshold:
            return True
        if not parent.is_evaluated or parent.train_nmse == float("inf"):
            return False
        return (
            child.train_nmse
            <= parent.train_nmse * DegenerationConfig().dominance_ratio
        )

    def _find_hiding_child(
        self,
        node: EvolutionNode,
    ) -> Optional[EvolutionNode]:
        pending = list(self.tree.get_children(node.skeleton_str))
        seen: Set[str] = set()
        while pending:
            child_formula = pending.pop(0)
            if child_formula in seen:
                continue
            seen.add(child_formula)
            child = self.tree.get_node(child_formula)
            if child is None:
                continue
            if self._child_hides_parent(node, child):
                return child
            if child.is_degenerated:
                pending.extend(self.tree.get_children(child_formula))
        return None

    def _mark_mature_nodes(self, log_new: bool = True) -> List[EvolutionNode]:
        """Collect current mature nodes and refresh transient mature cache."""
        visible_by_key: Dict[str, EvolutionNode] = {}
        visible: List[EvolutionNode] = []

        def _prefer_new(new: EvolutionNode, old: EvolutionNode) -> bool:
            new_complexity = (sum(new.operator_counts.values()), new.tree_depth)
            old_complexity = (sum(old.operator_counts.values()), old.tree_depth)
            if new_complexity != old_complexity:
                return new_complexity < old_complexity
            return new.train_nmse < old.train_nmse

        for node in self.tree.all_nodes:
            if node.is_degenerated:
                continue
            if not self._is_node_mature(node):
                continue

            hiding_child = self._find_hiding_child(node)
            if hiding_child is not None:
                node.mark_degenerated(
                    hiding_child.skeleton_str,
                    [
                        "hidden by sufficiently good child"
                    ],
                    kind="simplified",
                    child_train_nmse=hiding_child.train_nmse,
                    child_test_nmse=hiding_child.test_nmse,
                    child_ood_test_nmse=hiding_child.ood_test_nmse,
                )
                if node.skeleton_str in self._current_mature_formulas:
                    self._log(
                        f"  [Mature hidden] {node.skeleton_str}"
                        f" -> {hiding_child.skeleton_str}"
                    )
                continue

            key = node.canonical_key or self.normalizer.structural_key(
                node.skeleton_str)
            node.canonical_key = key
            existing = visible_by_key.get(key)
            if existing is not None:
                if _prefer_new(node, existing):
                    existing.mark_degenerated(
                        node.skeleton_str,
                        ["hidden by canonical mature duplicate"],
                        kind="duplicate",
                        child_train_nmse=node.train_nmse,
                        child_test_nmse=node.test_nmse,
                        child_ood_test_nmse=node.ood_test_nmse,
                    )
                    visible = [
                        v for v in visible
                        if v.skeleton_str != existing.skeleton_str
                    ]
                    visible_by_key[key] = node
                else:
                    node.mark_degenerated(
                        existing.skeleton_str,
                        ["hidden by canonical mature duplicate"],
                        kind="duplicate",
                        child_train_nmse=existing.train_nmse,
                        child_test_nmse=existing.test_nmse,
                        child_ood_test_nmse=existing.ood_test_nmse,
                    )
                    continue

            if node not in visible:
                visible.append(node)
            if log_new and node.skeleton_str not in self._reported_mature_formulas:
                self._log(
                    f"  [Mature] {node.skeleton_str}\n"
                    f"         train={self._fmt(node.train_nmse)}  "
                    f"test={self._fmt(node.test_nmse)}  "
                    f"ood={self._fmt(node.ood_test_nmse)}"
                )
                self._reported_mature_formulas.add(node.skeleton_str)

        self._current_mature_formulas = {node.skeleton_str for node in visible}
        return sorted(visible, key=lambda n: n.train_nmse)

    def _get_top_exprs(self, k: int = 3) -> List[Dict[str, Any]]:
        evaluated = [
            n for n in self.tree.all_nodes
            if n.is_evaluated and not n.is_degenerated
        ]
        top = sorted(evaluated, key=lambda n: n.train_nmse)[:k]
        results = []
        for n in top:
            d = {
                "expression": n.skeleton_str,
                "train_nmse": n.train_nmse,
            }
            params_str = n.get_fitted_params_str()
            if params_str:
                d["fitted_params"] = params_str
            results.append(d)
        return results

    def get_best_result(self) -> Dict[str, Any]:
        best = self.tree.best_node
        if best is None:
            return {}
        return {
            "expression": best.skeleton_str,
            "train_nmse": best.train_nmse,
            "test_nmse": best.test_nmse,
            "ood_test_nmse": best.ood_test_nmse,
            "params": best.param_names,
            "best_params": best.fitted_params,
            "description": best.description,
            "ast_features": sorted(best.ast_features),
        }

    @staticmethod
    def _fmt(val: float) -> str:
        if val == float("inf") or val >= 1e9:
            return "—"
        return f"{val:.2e}"

    def _log_unlocked(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        if self._log_file:
            self._log_file.write(msg + "\n")
            self._log_file.flush()

    def _log(self, msg: str) -> None:
        with self._lock:
            self._log_unlocked(msg)

    def _diagnostic_log_unlocked(self, msg: str) -> None:
        if self.verbose:
            print(msg)
        if self._diagnostic_log_file:
            self._diagnostic_log_file.write(msg + "\n")
            self._diagnostic_log_file.flush()

    def _diagnostic_log(self, msg: str) -> None:
        with self._lock:
            self._diagnostic_log_unlocked(msg)

    def _debug_log(self, msg: str) -> None:
        """Write verbose diagnostic details only when explicitly requested."""
        if self.verbose:
            self._log(msg)

    def close_log(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None
        if self._diagnostic_log_file:
            self._diagnostic_log_file.close()
            self._diagnostic_log_file = None


# Backward-compatible alias
EvolutionaryTreeSearch = TreeSearch
