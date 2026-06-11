# Unified Statistics Pipeline

`docs/scripts/build_statistics_pipeline.py` runs the whole statistics chain from
raw logs to every published CSV in one parallel command, replacing the old
sequence of four separately-invoked scripts.

```bash
conda run -n evolve python docs/scripts/build_statistics_pipeline.py --workers 16
```

## What it does (phases)

1. **Parse + rank (once, parallel).** `statistics.collect_statistics()` discovers
   and parses every `logs/<dataset>/...` file in one process pool and computes the
   ranker heuristics (global expression complexity once). The result is kept in
   memory instead of being thrown away after writing the workbook.
2. **`statistics.xlsx`.** `statistics.write_statistics_workbook()` serialises the
   in-memory structure (skip with `--skip-xlsx`).
3. **`docs/metrics/*.csv`.** For each dataset (clean + `noise1pct` + `noise5pct`)
   the per-cell NMSE / relative-error / Acc_tau are computed via
   `collect_selected_nmse.run_collect`, threading `--dataset` so each sheet is
   evaluated against the matching dataset variant.
4. **`docs/results/*.csv`.** Base reshape + ground truth + noisy tables +
   symbol-accuracy/ranker sync + the finite-filtered Acc_tau refresh
   (`--skip-results` to stop after metrics).

## Key design points

- **One dataset cache.** `docs/scripts/metrics_core.py` is the single source of
  truth for the `case -> split` mapping, a process-local `dataset_name`-aware
  `SRDataset` cache (`get_dataset`), expression evaluation, and the relative-error
  / Acc_tau / NMSE definitions. Each worker loads a given equation's split at most
  once across all the cells that reference it.
- **`dataset_name` is threaded** end to end. Note the noisy datasets only perturb
  `y_train`; `y_test`, `y_ood_test` and `X_train` are identical to the clean
  dataset, so test/OOD NMSE / rel-error / Acc are unchanged by the variant — the
  threading just makes the split selection explicit and correct.
- **Acc_tau** uses the finite-filtered pointwise definition: non-finite points
  are excluded, but a split with *no* finite points scores Acc = 0 (a failure
  that stays in the aggregate denominator, so totals remain out of 129). See
  `docs/results/README.md` and `docs/notes/acc_tau_relative_error_rerun.md`.

## Notes

- `statistics.py`, `collect_selected_nmse.py` and `build_all_results_csv.py`
  remain runnable on their own (CLIs unchanged); the orchestrator reuses their
  functions so outputs stay byte-compatible.
- `.tex` table generation (`generate_selected_nmse_detail_tables.py`) is not part
  of this pipeline; run it separately after refreshing the CSVs.
