# Results CSVs

This directory holds the final, self-contained per-experiment CSV exports used
for reporting. Each file is the curated output of the statistics pipeline; the
upstream metric caches, ranker sweeps, and generation scripts used to produce
them are kept on the development branch and are intentionally omitted from this
review snapshot.

## Common Columns

Each CSV expands the original `combined_status` column into:

- `SA@1`: symbol accuracy of the first candidate only.
- `SA@5`: symbol accuracy within the first 5 candidates.
- `SA@10`: symbol accuracy within the first 10 candidates.
- `SA@50`: symbol accuracy within the first 50 candidates.
- `SA@All`: symbol accuracy anywhere in the available candidate set.
- `pareto_occam@5`: symbol accuracy within the first 5 candidates ranked by
  the Pareto/Occam heuristic.
- `mdl@5`: symbol accuracy within the first 5 candidates ranked by the MDL
  heuristic.
- `pareto_occam@10`: symbol accuracy within the first 10 candidates ranked by
  the Pareto/Occam heuristic.
- `mdl@10`: symbol accuracy within the first 10 candidates ranked by the MDL
  heuristic.

All CSVs include strict Acc_tau columns (`test_Acc*`, `ood_Acc*`,
`both_Acc*`) and relaxed 95%Acc_tau columns (`test_95pctAcc*`,
`ood_95pctAcc*`, `both_95pctAcc*`). The relaxed version passes when at least
95% of evaluation points satisfy the corresponding relative-error tolerance.

Except for the baseline files listed below, `SA@k` columns are derived from the
rank-valued entries in the dataset-named sheets of `statistics.xlsx`:
positive ranks are hits for all thresholds greater than or equal to the rank,
while `0`, blank values, and `?` are non-hits. `Total_all` values are aligned
with the corresponding `statistics.xlsx` total rows.

Aggregate rows (`BPG_all`, `CRK_all`, `MatSci_all`, `PO_all`, and `Total_all`)
summarize individual cases in the same CSV: NMSE and max-relative-error columns
are medians for the corresponding split, while symbol-accuracy, heuristic
ranker, and Acc_tau columns are pass-counts formatted as `passed/total`.

## Ground Truth

File:

- `gt.csv`

Source:

- `datasets/llm-srbench/gt_expressions.csv`.

Rules:

- Contains the 129 individual test cases plus aggregate rows for `BPG_all`,
  `CRK_all`, `MatSci_all`, `PO_all`, and `Total_all`.
- `SA@1`, `SA@5`, `SA@10`, `SA@50`, and `SA@All` are set to `1` for every
  individual row; aggregate rows report pass-counts.
- Heuristic ranker columns are left blank because ground truth has no ranked
  candidate list.
- NMSE and max-relative-error columns are recomputed by evaluating the ground
  truth expression on each case's train / test / OOD arrays.

## Full

Files:

- `full_gpt5.2.csv`
- `full_llama3.1-8b.csv`
- `full_opus4.6.csv`
- `full_noise1pct_opus4.6.csv`
- `full_noise5pct_opus4.6.csv`

Symbol-accuracy ranks and totals come from the `llm-srbench` sheet of
`statistics.xlsx`. Noise files use the corresponding `statistics.xlsx` noise
sheets (`llm-srbench-noise1pct`, `llm-srbench-noise5pct`); their heuristic
columns are computed from the tuned Pareto/MDL parameters in `statistics.py`.

Current totals:

- `full_gpt5.2.csv`: `SA@1 = 52/129`, `SA@5 = 73/129`, `SA@10 = 80/129`,
  `SA@50 = 86/129`, `SA@All = 96/129`, `test_Acc0.01 = 87/129`,
  `test_95pctAcc0.01 = 115/129`
- `full_llama3.1-8b.csv`: `SA@1 = 23/129`, `SA@5 = 37/129`,
  `SA@10 = 48/129`, `SA@50 = 62/129`, `SA@All = 63/129`,
  `test_Acc0.01 = 30/129`, `test_95pctAcc0.01 = 59/129`
- `full_opus4.6.csv`: `SA@1 = 67/129`, `SA@5 = 80/129`, `SA@10 = 86/129`,
  `SA@50 = 99/129`, `SA@All = 112/129`, `test_Acc0.01 = 93/129`,
  `test_95pctAcc0.01 = 117/129`
- `full_noise1pct_opus4.6.csv`: `SA@1 = 24/129`, `SA@5 = 31/129`,
  `SA@10 = 37/129`, `SA@50 = 54/129`, `SA@All = 55/129`,
  `test_Acc0.01 = 14/129`, `test_95pctAcc0.01 = 51/129`
- `full_noise5pct_opus4.6.csv`: `SA@1 = 13/129`, `SA@5 = 26/129`,
  `SA@10 = 31/129`, `SA@50 = 40/129`, `SA@All = 41/129`,
  `test_Acc0.01 = 5/129`, `test_95pctAcc0.01 = 18/129`

## Baselines

Files:

- `llmsr_opus4.6.csv`
- `openevolve_opus4.6.csv`
- `direct_prompt_opus4.6.csv`

Rules:

- `matched_rank` is used as the match rank for `SA@1`, `SA@5`, `SA@10`,
  `SA@50`, and `SA@All`.
- Blank `matched_rank` means no symbolic-accuracy hit.
- Heuristic ranker columns are left blank because these baselines do not use
  the candidate-ranker heuristic outputs.

Current totals:

- `llmsr_opus4.6.csv`: `SA@1 = 11/129`, `SA@5 = 15/129`,
  `SA@10 = 18/129`, `SA@50 = 24/129`, `SA@All = 24/129`
- `openevolve_opus4.6.csv`: `SA@1 = 5/129`, `SA@5 = 13/129`,
  `SA@10 = 16/129`, `SA@50 = 24/129`, `SA@All = 24/129`
- `direct_prompt_opus4.6.csv`: `SA@1 = 1/129`, `SA@5 = 2/129`,
  `SA@10 = 2/129`, `SA@50 = 2/129`, `SA@All = 2/129`

## Ablations

Files:

- `degen_all_gpt5.2.csv`
- `degen_all_opus4.6.csv`
- `degen_generator_gpt5.2.csv`
- `degen_generator_opus4.6.csv`
- `degen_mutator1_gpt5.2.csv`
- `degen_mutator1_opus4.6.csv`
- `degen_mutator2_gpt5.2.csv`
- `degen_mutator2_opus4.6.csv`
- `degen_mutator3_selector2_opus4.6.csv`
- `degen_selector1_gpt5.2.csv`
- `degen_selector1_opus4.6.csv`
- `lbfgs_gpt5.2.csv`
- `lbfgs_opus4.6.csv`
- `only_structure_nollm.csv`

Symbol-accuracy ranks and totals come from `statistics.xlsx` (`llm-srbench`);
`only_structure` uses the `no-llm/only_structure` statistics column.

Selected current totals:

- `only_structure_nollm.csv`: `SA@1 = 16/129`, `SA@5 = 24/129`,
  `SA@10 = 25/129`, `SA@50 = 29/129`, `SA@All = 31/129`,
  `test_Acc0.01 = 53/129`, `test_95pctAcc0.01 = 89/129`
- `degen_generator_gpt5.2.csv`: `SA@1 = 49/129`, `SA@5 = 63/129`,
  `SA@10 = 72/129`, `SA@50 = 79/129`, `SA@All = 81/129`
- `degen_generator_opus4.6.csv`: `SA@1 = 58/129`, `SA@5 = 74/129`,
  `SA@10 = 81/129`, `SA@50 = 91/129`, `SA@All = 94/129`

## Acc_tau Definition

The Acc_tau columns use max relative error, not NMSE:

```text
rel_i         = |y_pred_i - y_i| / max(|y_i|, 1e-6)
valid_i       = isfinite(rel_i)
Acc_tau       = 1[max_{valid_i} rel_i <= tau]
95%Acc_tau    = 1[mean_{valid_i}(rel_i <= tau) >= 0.95]
```

Non-finite relative-error points are excluded from both strict and relaxed
Acc_tau calculations. A split with no finite evaluation points scores
Acc_tau = 0 (counted as a failure and kept in the aggregate denominator).

The source workflow is documented in
`docs/notes/acc_tau_relative_error_rerun.md`.
