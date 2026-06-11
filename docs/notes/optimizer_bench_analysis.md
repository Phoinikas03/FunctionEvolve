# Optimizer Benchmark Analysis

This note summarizes the optimizer benchmark in `symregression/optimizer_bench/`
and the results stored in `symregression/optimizer_bench/results/`.

## Purpose

The benchmark isolates the constant-fitting problem from symbolic search. It
takes ground-truth symbolic skeletons, replaces numeric constants with symbolic
parameters, applies controlled expression transformations, and then compares
whether different optimizers can recover parameters to very low error.

The success criterion used by `run_bench.py` and `plot_results.py` is
`final_nmse < 1e-10`.

## Case Construction

`bench_cases.py` reads ground-truth expressions from
`symregression/gt_expressions.csv`. Numeric constants are represented as
parameters `c0, c1, ...`; parameter products and coupled terms are decoupled
when possible so that linear coefficients can later be recovered more cleanly.
Training data are loaded from the local LLM-SRBench cache.

The benchmark creates one original case and six transformed variants for each
ground-truth expression:

| Phase | Transforms | Description | Purpose |
| --- | --- | --- | --- |
| T0 | `T0` | Original formula | Baseline fitting difficulty |
| T1 | `T1a`, `T1b` | Add composite zero terms, e.g. sinusoidal or exponential terms multiplied by zero-valued coefficients | Tests whether optimizers can suppress unnecessary nonlinear terms |
| T2 | `T2a`, `T2b` | Add power parameters to a feature or to the whole expression | Tests exponent search and flat/ill-conditioned objectives |
| T4 | `T4a`, `T4b` | Add rational zero terms | Tests rational expressions with removable nuisance terms |

The generated result set contains 903 cases across phases and 4515
case-optimizer records for five optimizers.

## Compared Optimizers

The benchmark compares the optimizers registered in `symregression/src/optimizer`:

| Optimizer | Implementation summary |
| --- | --- |
| `L-BFGS-B` | Multi-start local optimization over bounded parameters. |
| `least_squares` | Multi-start trust-region reflective residual minimization. |
| `CMA-ES` | Population-based global search with CMA-ES. |
| `DE` | Differential evolution with bounded global search. |
| `Structure` | A staged pipeline combining expression analysis, adaptive bounds, power pre-search, variable projection, TRF/CMA-ES/DE fallback, L-BFGS-B refinement, rational snapping, and power-exponent snapping/refit. |

The structure optimizer is not just an ensemble wrapper. It uses symbolic
structure to separate linear and nonlinear parameters when possible, solves
linear coefficients with OLS inside variable projection, and reserves global
search for the remaining nonlinear parameters. It also gives special treatment
to exponent parameters and negative-base power expressions.

## Protocol

`run_bench.py` evaluates each case-optimizer pair in a subprocess through
`ProcessPoolExecutor`. The default setting is three restarts and a 60 second
timeout per pair. Each record stores runtime, final MSE/NMSE, convergence
status, function evaluations, and timeout status.

## Results

The following table summarizes success rate and median runtime from
`results_all.csv`.

| Phase | Optimizer | Success | Success rate | Median time |
| --- | --- | ---: | ---: | ---: |
| T0 | Structure | 127/129 | 98.4% | 2.24s |
| T0 | CMA-ES | 52/129 | 40.3% | 3.26s |
| T0 | DE | 64/129 | 49.6% | 10.65s |
| T0 | L-BFGS-B | 10/129 | 7.8% | 0.17s |
| T0 | least_squares | 48/129 | 37.2% | 0.15s |
| T1 | Structure | 236/258 | 91.5% | 18.01s |
| T1 | CMA-ES | 85/258 | 32.9% | 13.26s |
| T1 | DE | 111/258 | 43.0% | 60.11s |
| T1 | L-BFGS-B | 17/258 | 6.6% | 0.63s |
| T1 | least_squares | 98/258 | 38.0% | 1.28s |
| T2 | Structure | 204/258 | 79.1% | 4.76s |
| T2 | CMA-ES | 26/258 | 10.1% | 9.79s |
| T2 | DE | 34/258 | 13.2% | 33.53s |
| T2 | L-BFGS-B | 3/258 | 1.2% | 0.12s |
| T2 | least_squares | 16/258 | 6.2% | 0.27s |
| T4 | Structure | 242/258 | 93.8% | 7.36s |
| T4 | CMA-ES | 83/258 | 32.2% | 9.18s |
| T4 | DE | 96/258 | 37.2% | 68.41s |
| T4 | L-BFGS-B | 14/258 | 5.4% | 0.68s |
| T4 | least_squares | 98/258 | 38.0% | 1.62s |

Overall, Structure solves 809/903 cases, corresponding to an 89.6% success rate.
The best non-structure optimizer overall is DE with 305/903 successful cases
(33.8%), but its median runtime is 44.42s and it frequently reaches the
timeout on transformed cases.

## Interpretation

The benchmark supports three conclusions.

First, reliable constant fitting is a major bottleneck for symbolic regression.
Even when the correct skeleton is known, local optimizers often fail to recover
constants to high precision.

Second, power transformations are the most difficult stress test. In T2,
Structure succeeds on 79.1% of cases, while the best non-structure optimizer reaches
only 13.2%. This gap is consistent with exponent parameters creating flat,
nonconvex, and discontinuity-prone objectives.

Third, the structure pipeline improves both robustness and efficiency. Pure global
search methods improve over local methods, but remain far below Structure and are
often slower. The strongest results come from combining symbolic decomposition,
global exploration, local refinement, and domain-specific handling of power and
rational parameters.

