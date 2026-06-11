#!/usr/bin/env python3
"""SA@1 first-appearance round, per run log.

For every log under ``logs/``, report the earliest search **Step** at which a candidate
that is *finally verified as symbolically equivalent to the ground truth* first appears.
If the log never achieves a symbolic match, the round label is ``no_match``.

Definition (as requested):
  1. Take the final verification verdict written into the log by ``verify.py``
     (``Matched candidate IDs: [...]`` / ``Conclusion: Match found`` / ``[MATCH FOUND]``).
  2. Map each matched candidate id to its expression via the last ``Final output`` block
     (``N. [Mature] <<<FORMULA>>> ... <<<END_FORMULA>>>``).
  3. Scan the log top-to-bottom tracking the current ``Step k/T`` section and find the
     first step at which each matched expression textually appears; the reported round is
     the minimum over all matched candidates.
  - Appearances in the seed / initialization section (before ``Step 1``) count as round 0.
  - ``0``  -> LLM-proposed seed / initialization candidate.
  - ``no_match`` -> no symbolic match (never found).
  - ``-1`` -> match found but the matched id could not be mapped to an expression
              (data issue; reported separately so coverage is auditable).

Scans both old ``logs/<task>/<model>/<date>/<eq>_<ts>.txt`` logs and the current
``logs/<benchmark>/<task>/<model>/<date>/<eq>_<ts>.txt`` layout (skips ``*legacy*``
model dirs and ``checkpoint/``). Writes ``sa_first_step.csv`` next to this script
(``docs/firstsa/sa_first_step.csv``).

Baseline methods (direct_prompt / llmsr / openevolve) use a different architecture
(no per-Step evolution tree), so the SA@1-round metric is not defined for them; they
are excluded by default. Pass ``--include-baselines`` to scan them too (their rows get
``source=baseline_arch`` and ``sa1_round`` is left blank).
"""
from __future__ import annotations

import os
import re
import csv
import argparse
from collections import defaultdict

BASE = os.path.dirname(os.path.abspath(__file__))
# This script lives in ``docs/firstsa/``; the run logs are at the repo root (two
# levels up). OUT_CSV stays next to the script (``docs/firstsa/sa_first_step.csv``).
ROOT = os.path.dirname(os.path.dirname(BASE))
LOGS_DIR = os.path.join(ROOT, "logs")
OUT_CSV = os.path.join(BASE, "sa_first_step.csv")

# Baseline tasks whose architecture has no evolution-tree "rounds"; SA@1-round is
# undefined for them, so they are not part of our ablation/full statistics.
BASELINE_TASKS = {"direct_prompt", "llmsr", "openevolve"}

# Candidate extraction mirrors verify.py::extract_candidates_from_file exactly: the
# matched candidate IDs are 1-based indices into the list of <<<FORMULA>>>...<<<END_FORMULA>>>
# matches over the WHOLE file (DOTALL), stripped, empties dropped. We must use the same
# ordering so an id maps to the same expression verify.py judged.
FORMULA_RE = re.compile(r"<<<FORMULA>>>(.+?)<<<END_FORMULA>>>", re.DOTALL)
MATCHED_IDS_RE = re.compile(
    r"Matched\s+candidate\s+IDs?\s*[:：]\s*\[([^\]\n]*)\]", re.IGNORECASE
)
STEP_RE = re.compile(r"^\s*Step\s+(\d+)\s*/\s*(\d+)")
DATE_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_matched_ids(text: str):
    matches = list(MATCHED_IDS_RE.finditer(text))
    if not matches:
        return None
    return [int(v) for v in re.findall(r"\d+", matches[-1].group(1))]


def final_match(text: str):
    """True / False / None for the log's final symbolic verdict."""
    ids = parse_matched_ids(text)
    if ids is not None:
        return len(ids) > 0
    for line in reversed(text.splitlines()):
        if "Conclusion:" in line:
            if "Match found" in line:
                return True
            if "No match" in line:
                return False
        if "[MATCH FOUND]" in line:
            return True
        if "[NO MATCH]" in line:
            return False
    return None


def extract_candidates(text):
    """1-based id -> candidate expression, exactly as verify.py would enumerate them."""
    cands = [m.strip() for m in FORMULA_RE.findall(text)]
    cands = [c for c in cands if c]
    return {i + 1: c for i, c in enumerate(cands)}


def extract_explicit_rounds(text):
    """1-based candidate id -> first-appearance round, from the Final output annotation.

    Only present in logs produced after the fix that stamps ``created_step``. The ids
    line up with ``extract_candidates`` because both enumerate the same FORMULA tags in
    order; we only collect annotations that sit on a FORMULA line (same ordering).
    Returns {} for legacy logs (no annotation), so callers can fall back to scanning.
    """
    rounds = {}
    idx = 0
    for m in FORMULA_RE.finditer(text):
        idx += 1  # 1-based id of this candidate
        tail = text[m.end():m.end() + 40]
        am = re.match(r"\s*\[first appeared: round (\w+)\]", tail)
        if am:
            tok = am.group(1)
            rounds[idx] = 0 if tok == "seed" else int(tok)
    return rounds


def get_max_steps(lines, text):
    m = re.search(r"max_steps\s*=\s*(\d+)", text)
    if m:
        return int(m.group(1))
    best = 0
    for ln in lines:
        s = STEP_RE.match(ln)
        if s:
            best = max(best, int(s.group(1)))
    return best


def _norm(s):
    """Whitespace-insensitive form for robust substring matching across log variants."""
    return re.sub(r"\s+", "", s)


def first_step_of_exprs(lines, exprs):
    """expr -> raw step index of first textual appearance (0 = seed / pre-Step-1).

    Step bodies print the expression after a ``→`` in the same normalized form used in
    the Final output block, so a whitespace-insensitive substring test locates the round
    the matched candidate was first produced.
    """
    targets = {e: _norm(e) for e in exprs if e}
    found = {}
    cur = 0
    for ln in lines:
        s = STEP_RE.match(ln)
        if s:
            cur = int(s.group(1))
            continue
        if not targets:
            break
        nl = _norm(ln)
        for e in [e for e, ne in targets.items() if ne in nl]:
            found[e] = cur
            del targets[e]
    return found


def search_process_lines(lines):
    """Return only the initialization/search portion, excluding final summaries.

    Legacy fallback matching is textual. If we scan into ``Final output`` or the
    verification block, every verified expression can appear there and be wrongly
    counted as a late search-round discovery.
    """
    out = []
    stop_prefixes = (
        "Search completed",
        "Final output",
        "Final search results",
        "GT Equivalence Verification",
    )
    for ln in lines:
        if any(ln.startswith(prefix) for prefix in stop_prefixes):
            break
        out.append(ln)
    return out


def analyze(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    lines = text.splitlines()
    ms = get_max_steps(lines, text)
    matched = final_match(text)
    if not matched:
        return dict(matched=0, sa1_round="no_match", n_matched=0, n_located=0, max_steps=ms,
                    source="no_match")
    ids = parse_matched_ids(text) or []

    # Preferred path: logs produced after the created_step fix carry an explicit
    # "[first appeared: round N]" annotation per candidate -> exact, no scanning.
    explicit = extract_explicit_rounds(text)
    if explicit:
        located = [explicit[i] for i in ids if i in explicit]
        if located:
            return dict(
                matched=1,
                sa1_round=min(located),
                n_matched=len(ids),
                n_located=len(located),
                max_steps=ms,
                source="annotation",
            )

    # Legacy path: map matched id -> expression, then scan Step bodies for it.
    id2expr = extract_candidates(text)
    targets = [id2expr[i] for i in ids if i in id2expr]
    if not targets:
        return dict(matched=1, sa1_round=-1, n_matched=len(ids), n_located=0,
                    max_steps=ms, source="none")
    first = first_step_of_exprs(search_process_lines(lines), targets)
    rounds = [first[e] for e in targets if e in first]
    return dict(
        matched=1,
        sa1_round=(min(rounds) if rounds else -1),
        n_matched=len(ids),
        n_located=len(rounds),
        max_steps=ms,
        source="scan",
    )


def main():
    ap = argparse.ArgumentParser(description="SA@1 first-appearance round per log.")
    ap.add_argument("--include-baselines", action="store_true",
                    help="Also scan baseline tasks (direct_prompt/llmsr/openevolve), "
                         "which have no evolution rounds (sa1_round left blank).")
    args = ap.parse_args()

    rows = []
    for root, dirs, files in os.walk(LOGS_DIR):
        dirs[:] = [
            d for d in dirs
            if d != "checkpoint" and "legacy" not in d.lower()
        ]
        rel_parts = os.path.relpath(root, LOGS_DIR).split(os.sep)
        if len(rel_parts) < 3 or not DATE_DIR_RE.match(rel_parts[-1]):
            continue
        model = rel_parts[-2]
        task = rel_parts[-3]
        benchmark = rel_parts[-4] if len(rel_parts) >= 4 else ""
        if "legacy" in model.lower():
            continue
        base_task = re.sub(r"_\d+$", "", task)
        is_baseline = base_task in BASELINE_TASKS
        if is_baseline and not args.include_baselines:
            continue
        for fn in sorted(files):
            if not fn.endswith((".txt", ".log")) or "_llm_usage_" in fn:
                continue
            fp = os.path.join(root, fn)
            eq = os.path.splitext(fn)[0].split("_")[0]
            if is_baseline:
                # Architecture has no rounds; record presence only.
                r = dict(matched="", sa1_round="", n_matched="",
                         n_located="", max_steps="", source="baseline_arch")
            else:
                try:
                    r = analyze(fp)
                except Exception as e:  # noqa: BLE001
                    r = dict(matched="ERR", sa1_round="", n_matched="",
                             n_located="", max_steps=str(e)[:60], source="err")
            rows.append(dict(benchmark=benchmark, model=model, task=task, base_task=base_task,
                             equation=eq, source_file=os.path.relpath(fp, ROOT), **r))

    fields = ["benchmark", "model", "task", "base_task", "equation", "matched", "sa1_round",
              "n_matched", "n_located", "max_steps", "source", "source_file"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    ours = [r for r in rows if r["source"] != "baseline_arch"]
    base = [r for r in rows if r["source"] == "baseline_arch"]
    tot = len(ours)
    matched = sum(1 for r in ours if r["matched"] == 1)
    located = sum(1 for r in ours if isinstance(r["sa1_round"], int) and r["sa1_round"] >= 0)
    unloc = sum(1 for r in ours if r["sa1_round"] == -1)
    no_match = sum(1 for r in ours if r["sa1_round"] == "no_match")
    print("=== SA@1 first-appearance round (ablation + full; baselines excluded) ===")
    print(f"logs(ours)={tot}  baselines_excluded={len(base)}")
    print(f"  no symbolic match (sa_first=no_match) : {no_match}")
    print(f"  matched & round located (sa_first>=0) : {located}")
    print(f"  matched but round unlocatable (-1)    : {unloc}")
    located_vals = [r["sa1_round"] for r in ours
                    if isinstance(r["sa1_round"], int) and r["sa1_round"] >= 0]
    if located_vals:
        sv = sorted(located_vals)
        mean = sum(sv) / len(sv)
        median = sv[len(sv) // 2] if len(sv) % 2 else (sv[len(sv) // 2 - 1] + sv[len(sv) // 2]) / 2
        le3 = sum(1 for v in sv if v <= 3)
        print(f"  located distribution: mean={mean:.2f} median={median} "
              f"min={sv[0]} max={sv[-1]} seed={sv.count(0)} round1={sv.count(1)} <=3={le3} "
              f"({100 * le3 / len(sv):.1f}%)")
    print()
    agg = defaultdict(list)
    for r in ours:
        if isinstance(r["sa1_round"], int) and r["sa1_round"] >= 0:
            agg[(r["model"], r["base_task"])].append(r["sa1_round"])
    for k in sorted(agg):
        v = agg[k]
        print(f"  {k[0]:14s} {k[1]:26s} n={len(v):4d} "
              f"mean_first_round={sum(v) / len(v):.2f} min={min(v)} max={max(v)}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
