# Acc_tau Relative-Error Rerun

This note documents the current rerun workflow after changing `Acc_tau` back to
its original pointwise relative-error definition.

## Metric

`docs/scripts/collect_selected_nmse.py` now computes:

```text
max_rel_error = max_i |y_pred_i - y_i| / max(|y_i|, 1e-6)
Acc_tau       = 1[max_rel_error <= tau]
```

The CSV still keeps NMSE columns, but `test_Acc0.1`, `test_Acc0.01`, and
`test_Acc0.001` are now generated from `test_max_rel_error`, not from
`test_nmse`. The same rule is used for OOD and both-split Acc columns.

## Environment

Use the `daily` conda environment from the repo root:

```bash
cd /Users/xia/Library/CloudStorage/OneDrive-个人/repos/functionevolve/symregression
conda activate daily
```

If a package is missing, install it into `daily`, for example:

```bash
pip install numpy scipy sympy openpyxl
```

## Inputs

The script expects these existing artifacts:

- `statistics.xlsx`
- `docs/metrics/direct_prompt_all_candidates_nmse.csv`
- `docs/metrics/selected_combined_nmse_best_params.csv`
- `logs/**` for FunctionEvolve and ablation runs
- `baseline/llm-srbench/logs/**` for LLM-SR
- `baseline/openevolve/examples/symbolic_regression/problems/**` for OpenEvolve

`selected_combined_nmse_best_params.csv` is used as a cache for expensive
LLM-SR/OpenEvolve fitted parameters. If a cached row is unavailable, the script
falls back to refitting parameters.

## Full Rerun

Run:

```bash
python docs/scripts/collect_selected_nmse.py --workers 8
python docs/scripts/generate_selected_nmse_detail_tables.py
```

Increase `--workers` on a larger machine if memory allows.

If the selected candidates and fitted parameters should be kept fixed, and only
the epsilon-stabilized relative-error / Acc_tau columns need to be refreshed,
run:

```bash
python docs/scripts/recompute_acc_tau_eps.py --eps 1e-6 --workers 48 --apply
```

## Split Rerun

For safer long runs, recompute one task column at a time and merge into the
existing CSV:

```bash
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task full --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_all --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_generator --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_selector1 --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_mutator1 --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_mutator2 --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task degen_mutator3_selector2 --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task lbfgs --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task only_hybrid --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task direct_prompt --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task llmsr --merge-existing
python docs/scripts/collect_selected_nmse.py --workers 8 --only-task openevolve --merge-existing
python docs/scripts/generate_selected_nmse_detail_tables.py
```

## Outputs

`collect_selected_nmse.py` writes:

- `docs/metrics/selected_combined_nmse.csv`
- `docs/metrics/openevolve_nmse.csv`
- `docs/metrics/selected_combined_nmse_summary.csv`

`generate_selected_nmse_detail_tables.py` writes:

- `docs/tables/selected_nmse_gpt52_details_table.tex`
- `docs/tables/selected_nmse_opus46_details_table.tex`
- `docs/tables/selected_nmse_baselines_details_table.tex`

After rerunning, update `neurips_2026.tex` tables from
`docs/metrics/selected_combined_nmse_summary.csv`. Do not compile LaTeX in this repo.
