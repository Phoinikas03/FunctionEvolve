#!/usr/bin/env python3
"""
Direct Prompting / DataBlind 基线：仅向 LLM 提供问题背景（symbol 描述）与变量名，
不包含任何数值数据；按固定次数调用 LLM（默认每题 1000 次），每次默认采样 5 个公式假设
采样温度取自 ``llm_config.yaml`` 的 ``generator.temperature``（缺省 0.8）。支持多线程并发调 API。
日志分两阶段：**阶段一**每次 LLM 调用后追加该次解析出的公式（纯文本行，无 <<<FORMULA>>>）；
**阶段二**对全部公式做**字符串去重**后，仅在该汇总块中用 <<<FORMULA>>> 包裹。含 equation=，
供根目录 verify.py 抽取全部候选后做一次 GT 等价性判断（verify 内部为单次判题 LLM 调用）。

CLI：`--dataset` 与 `--equation` **二选一**（不能同时出现）。
  --dataset：逗号分隔多个 split；对每个 split 内**全部**方程各跑一遍；
  --equation：逗号分隔多个方程名，用 find_equation_split 反查所属 split；
  --max-workers：题目级并发；每题内部仍有 --workers 控制采样并发。

用法（在仓库根目录）:
  export OPENAI_API_KEY=...
  python baseline/direct_prompt/run_direct_prompt.py --dataset bio_pop_growth --llm-config llm_config.yaml
  python baseline/direct_prompt/run_direct_prompt.py --dataset matsci,phys_osc --max-workers 4
  python baseline/direct_prompt/run_direct_prompt.py --equation lsr_synth_matsci_0 --verify

若使用本地代理:
  python baseline/local_server.py --port 8765
  python baseline/direct_prompt/run_direct_prompt.py --dataset bio_pop_growth \\
      --base-url http://127.0.0.1:8765/v1
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import json_repair

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.dataset import SRDataset, find_equation_split, set_repo_dir  # noqa: E402
from src.llm_client import build_openai_client, resolve_completion_max_tokens  # noqa: E402


def _extract_code_block(raw: str) -> str:
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    return m.group(1) if m else raw


def _parse_formula_json(raw: str) -> List[Dict[str, Any]]:
    text = _extract_code_block(raw)
    data = json_repair.loads(text)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def build_data_blind_prompt(ds: SRDataset) -> str:
    """仅背景 + 变量名，不含训练数据或样本数量。"""
    feat = ds.feature_names
    target = ds.symbols[0] if ds.symbols else "y"
    lines = [
        "## Symbolic regression task (data-free)",
        f"Task id: {ds.equation_name}",
        f"Dataset category: {ds.dataset_identifier}",
        f"Target variable (left-hand side / output): {target}",
        f"Feature variables (use only these names): {', '.join(feat)}",
    ]
    if ds.symbol_descs:
        lines.append("")
        lines.append("## Background (variable meanings)")
        for sym, desc in zip(ds.symbols, ds.symbol_descs):
            if desc:
                lines.append(f"- {sym}: {desc}")
    lines.append("")
    lines.append(
        "You must not use or assume access to numerical observations; "
        "propose formulas only from the background and variable names above."
    )
    return "\n".join(lines)


DATA_BLIND_SYSTEM = """You are a symbolic regression assistant.
Propose candidate formulas using SymPy-compatible syntax.
- Use only the variable names given in the user message.
- Represent unknown constants as c0, c1, c2, ... (multiplicative form preferred).
- Respond with a ```json code block only, no extra text outside the block.
"""


def _user_suffix(n_hypotheses: int) -> str:
    return f"""\
## Output format
Return a JSON array of exactly {n_hypotheses} objects in a ```json code block:
```json
[{{"expression": "<formula>", "params": ["c0", "c1", ...]}}, ...]
```
Each expression must be a single formula for the target variable in terms of the feature variables."""


class _ThreadLocalClient:
    """每个 worker 线程各自持有一个 LLM client，避免多线程共用一个 client。"""

    def __init__(
        self,
        gen_cfg: Dict[str, Any],
        base_url_override: Optional[str],
    ) -> None:
        self._gen = gen_cfg
        self._base_url_override = base_url_override
        self._local = threading.local()

    def get(self) -> Any:
        if getattr(self._local, "client", None) is None:
            model = self._gen.get("model", "")
            base_url = (
                self._base_url_override
                if self._base_url_override is not None
                else self._gen.get("base_url")
            )
            mode = self._gen.get("mode", "openai")
            self._local.client = build_openai_client(model, base_url, mode=mode)
        return self._local.client


def _one_llm_call(
    call_index: int,
    user_base: str,
    gen_cfg: Dict[str, Any],
    temperature: float,
    client_holder: _ThreadLocalClient,
    anthropic_version: Optional[str],
) -> Tuple[int, str, List[str]]:
    """单次 API 调用。返回 (index, raw_text, expression 列表)。"""
    model = gen_cfg.get("model", "")
    max_tokens = gen_cfg.get("max_tokens", 8192)
    client = client_holder.get()
    messages = [
        {"role": "system", "content": DATA_BLIND_SYSTEM},
        {"role": "user", "content": user_base},
    ]
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": resolve_completion_max_tokens(model, messages, max_tokens),
    }
    if anthropic_version:
        kwargs["anthropic_version"] = anthropic_version

    t0 = time.time()
    try:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        print(f"  [DirectPrompt] call {call_index + 1} failed: {e}")
        return (call_index, f"[ERROR] {e!s}", [])

    dt = time.time() - t0
    exprs: List[str] = []
    for p in _parse_formula_json(raw):
        expr = (p.get("expression") or "").strip()
        if expr and len(expr) <= 800:
            exprs.append(expr)
    print(
        f"  [DirectPrompt] call {call_index + 1} ok in {dt:.1f}s "
        f"(+{len(exprs)} parsed)"
    )
    return (call_index, raw, exprs)


def run_direct_prompt(
    ds: SRDataset,
    gen_cfg: Dict[str, Any],
    n_calls: int,
    hypotheses_per_call: int,
    temperature: float,
    base_url_override: Optional[str],
    workers: int,
    log_path: Optional[Path] = None,
    log_lock: Optional[threading.Lock] = None,
    save_raw_append: bool = False,
) -> tuple[List[str], List[str]]:
    """返回 (按 call 顺序、**字符串去重**后的公式列表, 按 call 下标对齐的 raw 文本列表)。

    若提供 ``log_path``，阶段一只追加无标签文本；阶段二由 ``write_log_final_section`` 写带标签汇总。
    """
    context = build_data_blind_prompt(ds)
    user_base = context + "\n\n" + _user_suffix(hypotheses_per_call)

    anthropic_version = gen_cfg.get("anthropic_version")
    client_holder = _ThreadLocalClient(gen_cfg, base_url_override)

    by_idx: Dict[int, Tuple[str, List[str]]] = {}
    workers = max(1, min(workers, n_calls))
    lock = log_lock or threading.Lock()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                _one_llm_call,
                i,
                user_base,
                gen_cfg,
                temperature,
                client_holder,
                anthropic_version,
            )
            for i in range(n_calls)
        ]
        for fut in as_completed(futs):
            idx, raw, exprs = fut.result()
            by_idx[idx] = (raw, exprs)
            if log_path is not None:
                append_log_llm_call(
                    log_path, lock, idx, raw, exprs, save_raw_append
                )

    seen: Set[str] = set()
    ordered: List[str] = []
    raw_chunks: List[str] = []
    for i in range(n_calls):
        raw, exprs = by_idx.get(i, ("", []))
        raw_chunks.append(raw)
        for expr in exprs:
            if expr not in seen:
                seen.add(expr)
                ordered.append(expr)

    print(
        f"  [DirectPrompt] done: {n_calls} calls, "
        f"{len(ordered)} unique formulas (string dedup)"
    )
    return ordered, raw_chunks


def write_log_header(
    path: Path,
    ds: SRDataset,
    n_calls: int,
    hypotheses_per_call: int,
    temperature: float,
    workers: int,
) -> None:
    """创建日志文件并写入元数据；随后按次追加 LLM 输出。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "symregression baseline: Direct Prompting (DataBlind)",
        "log_mode=phase1_plain_append_phase2_tagged_dedup",
        f"equation={ds.equation_name}",
        f"dataset={ds.dataset_identifier}",
        f"n_llm_calls={n_calls}",
        f"concurrent_workers={workers}",
        f"hypotheses_per_call={hypotheses_per_call}",
        f"temperature={temperature}",
        f"started={datetime.now().isoformat(timespec='seconds')}",
        "",
        "Phase 1 — per-call outputs (plain lines, no <<<FORMULA>>>; completion order):",
        "=" * 40,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_log_llm_call(
    path: Path,
    lock: threading.Lock,
    call_index: int,
    raw: str,
    exprs: List[str],
    save_raw: bool,
) -> None:
    """阶段一：单次调用完成后追加一段；公式为纯文本行，不使用 <<<FORMULA>>> 包裹。"""
    parts: List[str] = [f"\n--- llm_call {call_index + 1} ---\n"]
    for e in exprs:
        parts.append(f"{e}\n")
    if save_raw:
        parts.append("<raw>\n")
        parts.append(raw)
        parts.append("\n</raw>\n")
    block = "".join(parts)
    with lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)


def write_log_final_section(
    path: Path,
    formulas: List[str],
    raw_chunks: Optional[List[str]] = None,
) -> None:
    """阶段二：字符串去重后的公式列表，仅在此处用 <<<FORMULA>>> 包裹（供 verify 抽取）。"""
    with open(path, "a", encoding="utf-8") as f:
        f.write("\nPhase 2 — deduplicated formulas (string match), verify tags:\n")
        f.write("=" * 40 + "\n")
        for i, formula in enumerate(formulas, 1):
            f.write(f"{i}. {formula}\n")
            f.write(f"<<<FORMULA>>>{formula}<<<END_FORMULA>>>\n")
        f.write("=" * 40 + "\n")
        if raw_chunks:
            f.write("\nRAW_LLM_OUTPUTS_JSON\n")
            f.write(json.dumps(raw_chunks, ensure_ascii=False))
            f.write("\n")


def run_direct_prompt_single(
    split: str,
    equation: Optional[str],
    llm_config: Dict[str, Any],
    cfg_path: Path,
    base_url: Optional[str],
    n_calls: int,
    api_workers: int,
    hypotheses_per_call: int,
    save_raw: bool,
    do_verify: bool,
    output: Optional[str],
    verbose: bool = True,
) -> int:
    """
    跑单个 (split, equation)。equation=None 时加载该 split 中第一道方程。

    Returns
    -------
    0 成功；1 失败（含 verify 报错）。
    """
    gen_cfg = llm_config.get("generator")
    if not isinstance(gen_cfg, dict):
        print("Error: llm_config must contain a generator: mapping.", file=sys.stderr)
        return 1

    sample_temperature = float(gen_cfg.get("temperature", 0.8))

    ds = SRDataset.from_srbench(split, equation_name=equation)
    ds.load()

    out = output
    if not out:
        safe_name = ds.equation_name.replace("/", "_")
        out = str(
            _REPO_ROOT / "baseline" / "direct_prompt" / "logs" / f"{safe_name}_direct.txt"
        )

    if verbose:
        print(f"\n[DirectPrompt] === {ds.equation_name} ({split}) ===")

    out_path = Path(out)
    write_log_header(
        out_path,
        ds,
        n_calls=n_calls,
        hypotheses_per_call=hypotheses_per_call,
        temperature=sample_temperature,
        workers=api_workers,
    )
    log_lock = threading.Lock()
    formulas, raws = run_direct_prompt(
        ds,
        gen_cfg,
        n_calls=n_calls,
        hypotheses_per_call=hypotheses_per_call,
        temperature=sample_temperature,
        base_url_override=base_url,
        workers=api_workers,
        log_path=out_path,
        log_lock=log_lock,
        save_raw_append=save_raw,
    )
    write_log_final_section(
        out_path,
        formulas,
        raw_chunks=raws if save_raw else None,
    )
    print(f"[DirectPrompt] Wrote {len(formulas)} unique formulas to {out}")
    if not do_verify:
        print(f"[DirectPrompt] Verify with: python verify.py {out} --llm-config {cfg_path}")

    if do_verify:
        from verify import verify_log

        print("[DirectPrompt] Running verify.py equivalence check (single judge call)...")
        try:
            verify_text = verify_log(out, llm_config, verbose=verbose)
            with open(out, "a", encoding="utf-8") as f:
                f.write(verify_text + "\n")
            if verbose:
                print(f"[verify] Verification results appended to log: {out}")
            if verify_text.startswith("[verify error]"):
                return 1
        except Exception as e:
            print(f"[verify error] GT verification failed: {e}", file=sys.stderr)
            return 1

    return 0


def run_direct_prompt_all_in_split(
    split: str,
    llm_config: Dict[str, Any],
    cfg_path: Path,
    max_workers: int,
    base_url: Optional[str],
    n_calls: int,
    api_workers: int,
    hypotheses_per_call: int,
    save_raw: bool,
    do_verify: bool,
    verbose: bool = True,
) -> int:
    """对 split 内全部方程各跑一遍；题目级并发 max_workers。返回 0 当且仅当全部成功。"""
    ds_tmp = SRDataset.from_srbench(split)
    equations = ds_tmp.list_equations()
    if not equations:
        print(f"[DirectPrompt] No equations in split '{split}'.", file=sys.stderr)
        return 1

    n = len(equations)
    print(
        f"[DirectPrompt] Split '{split}': {n} equations, "
        f"equation-level workers={max_workers}"
    )

    any_fail = False

    def _job(eq: str) -> Tuple[str, int]:
        code = run_direct_prompt_single(
            split,
            eq,
            llm_config,
            cfg_path,
            base_url,
            n_calls,
            api_workers,
            hypotheses_per_call,
            save_raw,
            do_verify,
            output=None,
            verbose=verbose,
        )
        return eq, code

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        futures = {pool.submit(_job, eq): eq for eq in equations}
        for fut in as_completed(futures):
            eq = futures[fut]
            try:
                name, code = fut.result()
                if code != 0:
                    any_fail = True
                    print(f"[DirectPrompt] ✗ {name} (exit {code})", file=sys.stderr)
                else:
                    print(f"[DirectPrompt] ✓ {name}")
            except Exception as e:
                any_fail = True
                print(f"[DirectPrompt] ✗ {eq}: {e}", file=sys.stderr)

    return 1 if any_fail else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DataBlind direct prompting baseline (verify.py compatible logs)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Local path to the HF dataset repo (same as main.py --data-dir)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        metavar="SPLIT",
        help="srbench split (mutually exclusive with --equation). Comma-separated for multiple.\n"
        "Runs all equations in each split: bio_pop_growth | chem_react | matsci | phys_osc | lsrtransform",
    )
    parser.add_argument(
        "--equation",
        type=str,
        default=None,
        help="Comma-separated equation names (mutually exclusive with --dataset).\n"
        "Each name is resolved to a split via find_equation_split.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max concurrent equations when running multiple problems (default: 4)",
    )
    parser.add_argument("--list-equations", action="store_true",
                        help="List equation names for given --dataset and exit")
    parser.add_argument(
        "--llm-config",
        type=str,
        default=str(_REPO_ROOT / "llm_config.yaml"),
        help="YAML with generator section",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Override generator base_url (e.g. http://127.0.0.1:8765/v1 for local_server)",
    )
    parser.add_argument("--n-calls", type=int, default=1000, help="LLM API calls per problem")
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Concurrent API threads per problem (each thread uses its own client)",
    )
    parser.add_argument(
        "--hypotheses-per-call",
        type=int,
        default=5,
        help="How many formulas each response must list (paper: 5 in initial sampling)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Log file path (single-equation runs only; default: baseline/direct_prompt/logs/<equation>_direct.txt)",
    )
    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Append raw LLM outputs (large) to the log for debugging",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After each log, run root verify.py (one LLM judge call per problem)",
    )
    parser.add_argument("--quiet", action="store_true", help="Less console output")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.data_dir:
        set_repo_dir(args.data_dir)

    if args.dataset and args.equation:
        print(
            "Error: specify exactly one of --dataset or --equation (not both).",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.list_equations:
        if not args.dataset:
            print("Please specify --dataset, e.g.: --dataset bio_pop_growth", file=sys.stderr)
            sys.exit(1)
        ds_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
        for ds_name in ds_names:
            if len(ds_names) > 1:
                print(f"\n=== {ds_name} ===")
            ds_tmp = SRDataset.from_srbench(ds_name)
            for eq in ds_tmp.list_equations():
                print(f"  {eq}")
        sys.exit(0)

    import yaml

    cfg_path = Path(args.llm_config)
    if not cfg_path.is_file():
        print(f"Error: llm config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        llm_config = yaml.safe_load(f) or {}

    verbose = not args.quiet

    if args.equation:
        eq_names = [e.strip() for e in args.equation.split(",") if e.strip()]

        eq_split_map: List[Tuple[str, str]] = []
        for eq in eq_names:
            if verbose:
                print(f"[DirectPrompt] {eq} -> ", end="", flush=True)
            split = find_equation_split(eq)
            if split is None:
                print("not found, skip." if verbose else "")
                continue
            if verbose:
                print(split)
            eq_split_map.append((eq, split))

        if not eq_split_map:
            print("No valid equations.", file=sys.stderr)
            sys.exit(1)

        if len(eq_split_map) == 1:
            eq, split = eq_split_map[0]
            code = run_direct_prompt_single(
                split,
                eq,
                llm_config,
                cfg_path,
                args.base_url,
                args.n_calls,
                args.workers,
                args.hypotheses_per_call,
                args.save_raw,
                args.verify,
                args.output,
                verbose=verbose,
            )
            sys.exit(code)

        if args.output:
            print(
                "[DirectPrompt] --output ignored when running multiple equations.",
                file=sys.stderr,
            )

        any_fail = False

        def _run_mapped(item: Tuple[str, str]) -> Tuple[str, int]:
            eq, split = item
            c = run_direct_prompt_single(
                split,
                eq,
                llm_config,
                cfg_path,
                args.base_url,
                args.n_calls,
                args.workers,
                args.hypotheses_per_call,
                args.save_raw,
                args.verify,
                output=None,
                verbose=verbose,
            )
            return eq, c

        print(
            f"\n[DirectPrompt] {len(eq_split_map)} equations, "
            f"equation-level workers={args.max_workers}\n"
        )
        with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as pool:
            futures = {pool.submit(_run_mapped, item): item[0] for item in eq_split_map}
            for fut in as_completed(futures):
                eq = futures[fut]
                try:
                    name, code = fut.result()
                    if code != 0:
                        any_fail = True
                        print(f"[DirectPrompt] ✗ {name} (exit {code})", file=sys.stderr)
                    else:
                        print(f"[DirectPrompt] ✓ {name}")
                except Exception as e:
                    any_fail = True
                    print(f"[DirectPrompt] ✗ {eq}: {e}", file=sys.stderr)

        sys.exit(1 if any_fail else 0)

    if args.dataset:
        ds_names = [d.strip() for d in args.dataset.split(",") if d.strip()]
        any_fail = False
        for ds_name in ds_names:
            if len(ds_names) > 1:
                print(f"\n########## split: {ds_name} ##########")
            code = run_direct_prompt_all_in_split(
                ds_name,
                llm_config,
                cfg_path,
                args.max_workers,
                args.base_url,
                args.n_calls,
                args.workers,
                args.hypotheses_per_call,
                args.save_raw,
                args.verify,
                verbose=verbose,
            )
            if code != 0:
                any_fail = True
        sys.exit(1 if any_fail else 0)

    print(
        "Error: specify exactly one of --dataset or --equation (or use --list-equations with --dataset).\n"
        "  Examples:\n"
        "    python baseline/direct_prompt/run_direct_prompt.py --dataset matsci\n"
        "    python baseline/direct_prompt/run_direct_prompt.py --equation lsr_synth_matsci_0",
        file=sys.stderr,
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
