# Docs Reorganization Manifest

This manifest records the reorganization that moved the previous top-level
files in `docs/` into purpose-specific subdirectories. In the move table,
`Current path` means the pre-migration path and `Proposed path` is the path now
used in this workspace.

## Target Directories

| Directory | Purpose |
|---|---|
| `docs/results/` | Final per-experiment CSVs for reporting. Already exists. |
| `docs/metrics/` | NMSE, Acc_tau, best-parameter, and candidate metric tables. |
| `docs/verification/` | Symbol-accuracy, candidate-1, LLM-judge, and GT-match artifacts. |
| `docs/figures/` | Rendered figure assets: PNG, PDF, SVG. |
| `docs/figure_data/` | CSVs used directly as figure data. |
| `docs/scripts/` | Non-plot data extraction, metric collection, and table-generation scripts. |
| `docs/plots/` | Plotting scripts. |
| `docs/tables/` | Generated LaTeX tables. |
| `docs/notes/` | Human-readable notes, explanations, and analysis writeups. |
| `docs/llm_usage/` | Existing LLM usage accounting folder. Keep in place. |
| `docs/top50_symbol_accuracy/` | Existing top-50 symbol-accuracy folder. Keep in place. |
| `docs/verify_audit/` | Existing verification audit folder. Keep in place. |

## Move Plan

| Current path | Proposed path | Notes |
|---|---|---|
| `docs/AUTO_VERIFICATION_CHECK.md` | `docs/notes/AUTO_VERIFICATION_CHECK.md` | Documentation only. |
| `docs/FIX_SUMMARY.md` | `docs/notes/FIX_SUMMARY.md` | Documentation only. |
| `docs/LLM_JUDGE_INPUT_INFO.md` | `docs/notes/LLM_JUDGE_INPUT_INFO.md` | Documentation only. |
| `docs/MATCH_DETECTION_MECHANISM.md` | `docs/notes/MATCH_DETECTION_MECHANISM.md` | Documentation only. |
| `docs/REASONING_EFFORT_GUIDE.md` | `docs/notes/REASONING_EFFORT_GUIDE.md` | Documentation only. |
| `docs/REASONING_PARAMETER_CHECK.md` | `docs/notes/REASONING_PARAMETER_CHECK.md` | Documentation only. |
| `docs/acc_tau_relative_error_rerun.md` | `docs/notes/acc_tau_relative_error_rerun.md` | Update links from `docs/results/README.md` if moved. |
| `docs/collect_selected_nmse.py` | `docs/scripts/collect_selected_nmse.py` | Requires path updates; many output paths currently assume `docs/`. |
| `docs/direct_prompt_all_candidates_best_params.json` | `docs/metrics/direct_prompt_all_candidates_best_params.json` | Input for selected metric collection. Update script path if moved. |
| `docs/direct_prompt_all_candidates_nmse.csv` | `docs/metrics/direct_prompt_all_candidates_nmse.csv` | Input for selected metric collection. Update script path if moved. |
| `docs/extract_gpt52_full_candidate1_trace.py` | `docs/scripts/extract_gpt52_full_candidate1_trace.py` | Requires output path updates. |
| `docs/extract_opus46_full_candidate1_trace.py` | `docs/scripts/extract_opus46_full_candidate1_trace.py` | Requires output path updates. |
| `docs/extract_opus46_full_coefficient_solutions.py` | `docs/scripts/extract_opus46_full_coefficient_solutions.py` | Requires input/output path updates. |
| `docs/extract_opus46_full_gt_matches.py` | `docs/scripts/extract_opus46_full_gt_matches.py` | Requires output path updates. |
| `docs/generate_selected_nmse_detail_tables.py` | `docs/scripts/generate_selected_nmse_detail_tables.py` | Requires input/output path updates. |
| `docs/gpt52_full_candidate1_sa.csv` | `docs/verification/gpt52_full_candidate1_sa.csv` | Source for `results/full_gpt5.2.csv`. Update results README and generation script if moved. |
| `docs/gpt52_full_candidate1_trace_from_verify.csv` | `docs/verification/gpt52_full_candidate1_trace_from_verify.csv` | Source for `results/full_gpt5.2.csv`. |
| `docs/gt_nmse_distribution.pdf` | `docs/figures/gt_nmse_distribution.pdf` | Figure artifact. |
| `docs/gt_nmse_distribution.png` | `docs/figures/gt_nmse_distribution.png` | Figure artifact. |
| `docs/gt_nmse_distribution_hist.pdf` | `docs/figures/gt_nmse_distribution_hist.pdf` | Figure artifact. |
| `docs/gt_nmse_distribution_hist.png` | `docs/figures/gt_nmse_distribution_hist.png` | Figure artifact. |
| `docs/gt_nmse_distribution_summary.csv` | `docs/figure_data/gt_nmse_distribution_summary.csv` | Figure data. |
| `docs/main_workflow_core_components.md` | `docs/notes/main_workflow_core_components.md` | Documentation only. |
| `docs/matsci_data_degeneracy_analysis.md` | `docs/notes/matsci_data_degeneracy_analysis.md` | Documentation only. |
| `docs/matsci_input_degeneracy.pdf` | `docs/figures/matsci_input_degeneracy.pdf` | Figure artifact. |
| `docs/matsci_input_degeneracy.png` | `docs/figures/matsci_input_degeneracy.png` | Figure artifact. |
| `docs/matsci_input_residual.pdf` | `docs/figures/matsci_input_residual.pdf` | Figure artifact. |
| `docs/matsci_input_residual.png` | `docs/figures/matsci_input_residual.png` | Figure artifact. |
| `docs/matsci_input_samples.pdf` | `docs/figures/matsci_input_samples.pdf` | Figure artifact. |
| `docs/matsci_input_samples.png` | `docs/figures/matsci_input_samples.png` | Figure artifact. |
| `docs/only_hybrid_candidate_nmse_PO36_PO37_PO42_PO43.csv` | `docs/metrics/only_hybrid_candidate_nmse_PO36_PO37_PO42_PO43.csv` | Candidate metric audit. |
| `docs/openevolve_nmse.csv` | `docs/metrics/openevolve_nmse.csv` | Derived metric subset. |
| `docs/optimizer.md` | `docs/notes/optimizer.md` | Documentation only. |
| `docs/optimizer_bench_analysis.md` | `docs/notes/optimizer_bench_analysis.md` | Documentation only. |
| `docs/opus46_full_candidate1_sa.csv` | `docs/verification/opus46_full_candidate1_sa.csv` | Source for `results/full_opus4.6.csv`. |
| `docs/opus46_full_candidate1_trace_from_verify.csv` | `docs/verification/opus46_full_candidate1_trace_from_verify.csv` | Source for `results/full_opus4.6.csv`. |
| `docs/opus46_full_gt_coefficient_solutions.csv` | `docs/verification/opus46_full_gt_coefficient_solutions.csv` | GT-match coefficient analysis. |
| `docs/opus46_full_gt_match_coefficients.csv` | `docs/verification/opus46_full_gt_match_coefficients.csv` | GT-match coefficient analysis. |
| `docs/opus46_full_gt_match_table.tex` | `docs/tables/opus46_full_gt_match_table.tex` | Generated table. |
| `docs/opus46_full_nmse_figures.md` | `docs/notes/opus46_full_nmse_figures.md` | Figure documentation. Update figure paths if moved. |
| `docs/opus46_full_ood_nmse_combined.pdf` | `docs/figures/opus46_full_ood_nmse_combined.pdf` | Figure artifact. |
| `docs/opus46_full_ood_nmse_combined.png` | `docs/figures/opus46_full_ood_nmse_combined.png` | Figure artifact. |
| `docs/opus46_full_ood_nmse_delta_ecdf.pdf` | `docs/figures/opus46_full_ood_nmse_delta_ecdf.pdf` | Figure artifact. |
| `docs/opus46_full_ood_nmse_delta_ecdf.png` | `docs/figures/opus46_full_ood_nmse_delta_ecdf.png` | Figure artifact. |
| `docs/opus46_full_ood_nmse_delta_values.csv` | `docs/figure_data/opus46_full_ood_nmse_delta_values.csv` | Figure data. |
| `docs/opus46_full_ood_nmse_delta_vs_gt.pdf` | `docs/figures/opus46_full_ood_nmse_delta_vs_gt.pdf` | Figure artifact. |
| `docs/opus46_full_ood_nmse_delta_vs_gt.png` | `docs/figures/opus46_full_ood_nmse_delta_vs_gt.png` | Figure artifact. |
| `docs/opus46_full_ood_nmse_final_vs_gt.csv` | `docs/figure_data/opus46_full_ood_nmse_final_vs_gt.csv` | Figure data. |
| `docs/opus46_full_ood_nmse_final_vs_gt.pdf` | `docs/figures/opus46_full_ood_nmse_final_vs_gt.pdf` | Figure artifact. |
| `docs/opus46_full_ood_nmse_final_vs_gt.png` | `docs/figures/opus46_full_ood_nmse_final_vs_gt.png` | Figure artifact. |
| `docs/opus46_full_ood_nmse_log_quantiles.csv` | `docs/figure_data/opus46_full_ood_nmse_log_quantiles.csv` | Figure data. |
| `docs/opus46_full_ood_nmse_log_quantiles.pdf` | `docs/figures/opus46_full_ood_nmse_log_quantiles.pdf` | Figure artifact. |
| `docs/opus46_full_ood_nmse_log_quantiles.png` | `docs/figures/opus46_full_ood_nmse_log_quantiles.png` | Figure artifact. |
| `docs/opus46_full_test_nmse_combined.pdf` | `docs/figures/opus46_full_test_nmse_combined.pdf` | Figure artifact. |
| `docs/opus46_full_test_nmse_combined.png` | `docs/figures/opus46_full_test_nmse_combined.png` | Figure artifact. |
| `docs/opus46_full_test_nmse_delta_ecdf.pdf` | `docs/figures/opus46_full_test_nmse_delta_ecdf.pdf` | Figure artifact. |
| `docs/opus46_full_test_nmse_delta_ecdf.png` | `docs/figures/opus46_full_test_nmse_delta_ecdf.png` | Figure artifact. |
| `docs/opus46_full_test_nmse_delta_values.csv` | `docs/figure_data/opus46_full_test_nmse_delta_values.csv` | Figure data. |
| `docs/opus46_full_test_nmse_delta_vs_gt.pdf` | `docs/figures/opus46_full_test_nmse_delta_vs_gt.pdf` | Figure artifact. |
| `docs/opus46_full_test_nmse_delta_vs_gt.png` | `docs/figures/opus46_full_test_nmse_delta_vs_gt.png` | Figure artifact. |
| `docs/opus46_full_test_nmse_final_vs_gt.csv` | `docs/figure_data/opus46_full_test_nmse_final_vs_gt.csv` | Figure data. |
| `docs/opus46_full_test_nmse_final_vs_gt.pdf` | `docs/figures/opus46_full_test_nmse_final_vs_gt.pdf` | Figure artifact. |
| `docs/opus46_full_test_nmse_final_vs_gt.png` | `docs/figures/opus46_full_test_nmse_final_vs_gt.png` | Figure artifact. |
| `docs/opus46_full_test_nmse_log_quantiles.csv` | `docs/figure_data/opus46_full_test_nmse_log_quantiles.csv` | Figure data. |
| `docs/opus46_full_test_nmse_log_quantiles.pdf` | `docs/figures/opus46_full_test_nmse_log_quantiles.pdf` | Figure artifact. |
| `docs/opus46_full_test_nmse_log_quantiles.png` | `docs/figures/opus46_full_test_nmse_log_quantiles.png` | Figure artifact. |
| `docs/plot_gt_nmse_distribution.py` | `docs/plots/plot_gt_nmse_distribution.py` | Requires output path updates. |
| `docs/plot_gt_nmse_distribution_hist.py` | `docs/plots/plot_gt_nmse_distribution_hist.py` | Requires output path updates. |
| `docs/plot_matsci_input_degeneracy.py` | `docs/plots/plot_matsci_input_degeneracy.py` | Requires output path updates. |
| `docs/plot_opus46_full_combined_nmse.py` | `docs/plots/plot_opus46_full_combined_nmse.py` | Requires imports/path updates if moved. |
| `docs/plot_opus46_full_delta_views.py` | `docs/plots/plot_opus46_full_delta_views.py` | Requires imports/path updates if moved. |
| `docs/plot_opus46_full_gt_scatter.py` | `docs/plots/plot_opus46_full_gt_scatter.py` | Requires imports/path updates if moved. |
| `docs/plot_opus46_full_quantiles.py` | `docs/plots/plot_opus46_full_quantiles.py` | Requires imports/path updates if moved. |
| `docs/run_verify_audit.py` | `docs/scripts/run_verify_audit.py` | Requires path updates. |
| `docs/run_verify_audit.sh` | `docs/scripts/run_verify_audit.sh` | Requires path updates. |
| `docs/search_flow_core_components.svg` | `docs/figures/search_flow_core_components.svg` | Figure artifact. |
| `docs/selected_combined_nmse.csv` | `docs/metrics/selected_combined_nmse.csv` | Main metrics table. Many scripts/readmes reference this path. |
| `docs/selected_combined_nmse.md` | `docs/notes/selected_combined_nmse.md` | Documentation only; update links if needed. |
| `docs/selected_combined_nmse_best_params.csv` | `docs/metrics/selected_combined_nmse_best_params.csv` | Metric parameter table. |
| `docs/selected_combined_nmse_best_params.json` | `docs/metrics/selected_combined_nmse_best_params.json` | Metric parameter table. |
| `docs/selected_combined_nmse_summary.csv` | `docs/metrics/selected_combined_nmse_summary.csv` | Summary metrics table. |
| `docs/selected_nmse_baselines_details_table.tex` | `docs/tables/selected_nmse_baselines_details_table.tex` | Generated table. |
| `docs/selected_nmse_gpt52_details_table.tex` | `docs/tables/selected_nmse_gpt52_details_table.tex` | Generated table. |
| `docs/selected_nmse_opus46_details_table.tex` | `docs/tables/selected_nmse_opus46_details_table.tex` | Generated table. |
| `docs/statistics_status_traceability.csv` | `docs/verification/statistics_status_traceability.csv` | Traceability/audit artifact. |
| `docs/statistics_xlsx_traceability.md` | `docs/notes/statistics_xlsx_traceability.md` | Documentation only. |
| `docs/top50_direct_prompt.csv` | `docs/metrics/top50_direct_prompt.csv` | Raw top-candidate baseline candidates. Consider moving into `top50_symbol_accuracy/` if preferred. |
| `docs/top50_llmsr.csv` | `docs/metrics/top50_llmsr.csv` | Raw top-candidate baseline candidates. Consider moving into `top50_symbol_accuracy/` if preferred. |
| `docs/top50_openevolve.csv` | `docs/metrics/top50_openevolve.csv` | Raw top-candidate baseline candidates. Consider moving into `top50_symbol_accuracy/` if preferred. |
| `docs/tree_node.md` | `docs/notes/tree_node.md` | Documentation only. |
| `docs/verify_audit_llm_judge_reliability.md` | `docs/notes/verify_audit_llm_judge_reliability.md` | Documentation only. |
| `docs/verify_gpt52_full_candidate1_sa.py` | `docs/scripts/verify_gpt52_full_candidate1_sa.py` | Requires input/output path updates. |
| `docs/verify_opus46_full_candidate1_sa.py` | `docs/scripts/verify_opus46_full_candidate1_sa.py` | Requires input/output path updates. |

## Existing Directories To Keep

| Current path | Proposed action |
|---|---|
| `docs/results/` | Keep. Already contains final per-experiment CSVs and source README. |
| `docs/llm_usage/` | Keep. Already internally organized around LLM usage accounting. |
| `docs/top50_symbol_accuracy/` | Keep. Contains top-50 SA verification outputs. |
| `docs/verify_audit/` | Keep. Contains audit samples and audit result subdirectories. |

## Cleanup Candidates

| Path | Suggested action |
|---|---|
| `docs/__pycache__/` | Remove from repository/worktree if not needed. Generated Python cache. |

## Path Update Hotspots

The migration updated hard-coded paths in these scripts:

- `docs/scripts/collect_selected_nmse.py`
- `docs/scripts/generate_selected_nmse_detail_tables.py`
- `docs/scripts/verify_gpt52_full_candidate1_sa.py`
- `docs/scripts/verify_opus46_full_candidate1_sa.py`
- `docs/scripts/extract_gpt52_full_candidate1_trace.py`
- `docs/scripts/extract_opus46_full_candidate1_trace.py`
- `docs/scripts/extract_opus46_full_gt_matches.py`
- `docs/scripts/extract_opus46_full_coefficient_solutions.py`
- `docs/plots/plot_gt_nmse_distribution.py`
- `docs/plots/plot_gt_nmse_distribution_hist.py`
- `docs/plots/plot_matsci_input_degeneracy.py`
- `docs/plots/plot_opus46_full_combined_nmse.py`
- `docs/plots/plot_opus46_full_delta_views.py`
- `docs/plots/plot_opus46_full_gt_scatter.py`
- `docs/plots/plot_opus46_full_quantiles.py`
- `docs/scripts/run_verify_audit.py`
- `docs/scripts/run_verify_audit.sh`

Recommended migration order:

1. Create the target directories.
2. Move documentation-only files first (`notes/`, `figures/`, `figure_data/`,
   `tables/`).
3. Move metric and verification CSV/JSON artifacts.
4. Move scripts last, after path constants are updated.
5. Re-run the key generation commands and compare outputs.
