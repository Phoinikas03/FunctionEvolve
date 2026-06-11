# statistics.xlsx status traceability

This note documents how each `0`, `1`, and `?` status cell in `../statistics.xlsx` was traced back through `../statistics.py` to the current files under `../logs`.

## Source logic confirmed from statistics.py

- Source layout: `logs/<dataset>/<task_name>/<model_name>/<filename>.txt|.log`.
- Model folders containing `legacy` are ignored case-insensitively.
- `test_case` is the filename stem before the first underscore. Example: `BPG0_20260407_120000.txt` becomes `BPG0`.
- File status is derived from content:
  - `1`: contains `Conclusion: Match found` and not `Conclusion: No match`.
  - `0`: contains `Conclusion: No match` and not `Conclusion: Match found`.
  - `?`: contains both conclusion strings.
  - blank: contains neither conclusion string.
- `Original Statistics` keeps the last readable file encountered for each exact `(test_case, model_name, task_name)`, where traversal sorts `filename` inside each model folder.
- `Combined Statistics` removes a trailing `_<number>` from the task name, then merges all raw readable file statuses for the same `(test_case, model_name, base_task_name)` using priority `1 > ? > 0 > blank`.

## Generated index

The full per-cell mapping is in `docs/verification/statistics_status_traceability.csv`.

Important columns:

- `sheet`, `xlsx_row`, `xlsx_column`: location in `statistics.xlsx`.
- `test_case`, `model`, `task_or_combined_task`, `xlsx_status`: the visible xlsx cell meaning.
- `statistics_py_status_from_current_logs`: the status reproduced from current logs using the same logic.
- `source_file`: readable source file for confirmed rows, or the current unreadable candidate path for unresolved symlink rows.
- `trace_status`: whether the cell was confirmed or why it remains unresolved.

## Verification summary

| Sheet | Status cells in xlsx | Confirmed from current readable logs | Unresolved against current logs | Mismatched vs rerun logic |
|---|---:|---:|---:|---:|
| Combined Statistics | 2322 | 1935 | 387 | 387 |
| Original Statistics | 2615 | 2228 | 387 | 387 |

All unresolved cells are in these `opus-4-6` columns: `direct_prompt`, `llmsr`, and `openevolve`.

- `direct_prompt`: the current `logs/direct_prompt/opus-4-6` tree contains no `.txt` or `.log` files, so the xlsx `0` values cannot be confirmed from current files.
- `llmsr` and `openevolve`: the current entries are symlinks, but their targets are not readable from this workspace. `statistics.py` would catch the read error and skip them, so the xlsx `0` values cannot be reproduced from the current workspace state.

## Reading combined rows

For `Combined Statistics`, one xlsx cell can map to multiple rows in the CSV. That is intentional: a combined `1` corresponds to every readable source file with status `1` that contributes under the merge rule, while a combined `0` corresponds to readable `0` sources only when no higher-priority `1` or `?` source exists.
