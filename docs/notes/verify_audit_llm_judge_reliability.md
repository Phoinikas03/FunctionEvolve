# LLM-as-a-Judge Reliability Audit

This audit summarizes the 500 sampled verification cases in
`docs/verify_audit/sample_500_seed20260502.csv`, judged once by GPT-5.2 and once
by Claude Opus 4.6. When the two judges disagreed, or when either run produced
`verify_error`, the case was manually adjudicated.

## Evaluation protocol

- Total audited samples: 500
- Judges:
  - GPT-5.2: `gpt-5.2-2025-12-11`
  - Claude Opus 4.6: `claude-opus-4-6`
- Judge outputs were reduced to `match`, `no_match`, or `verify_error`.
- For Opus 4.6, repeated rows caused by retries were deduplicated by
  `sample_id`; the final non-`verify_error` result was used when available.
- The manual adjudication uses the same rule as `verify.py`: GT constants are
  fixed, candidate constants are free, and PO tasks allow negative variables, so
  `v**c` is not treated as equivalent to `Abs(v)**c`.

## Agreement statistics

| Category | Count | Rate |
|---|---:|---:|
| Total samples | 500 | 100.00% |
| Same valid verdict (`match`/`no_match`) | 476 | 95.20% |
| Valid verdict disagreement | 21 | 4.20% of all samples; 4.23% of 497 comparable samples |
| Cases involving `verify_error` | 3 | 0.60% |
| Manual intervention cases | 24 | 4.80% |
| Unscorable case after audit | 1 | 0.20% |

There were 23 status-level differences if `verify_error` is counted as a status:
21 valid verdict disagreements plus 2 cases where GPT-5.2 returned
`verify_error` and Opus 4.6 returned a valid verdict. One additional case
(`sample_id=294`) had `verify_error` from both judges and could not be scored
because no final candidate formulas were extractable from the log.

## Accuracy after manual adjudication

The 476 same-verdict cases were accepted as correct under the two-judge protocol.
The remaining 24 cases were manually checked; 23 had a valid final label and 1
was unscorable.

| Judge | Correct on 499 scorable samples | Accuracy on 499 scorable samples | Strict correct / 500 | Strict accuracy / 500 |
|---|---:|---:|---:|---:|
| GPT-5.2 | 482 | 96.59% | 482 | 96.40% |
| Claude Opus 4.6 | 493 | 98.80% | 493 | 98.60% |

On the 23 manually scorable intervention cases, GPT-5.2 was correct on 6 and
Opus 4.6 was correct on 17. The only unscorable case was `sample_id=294`, where
both judges failed because the log contained no extractable final formulas.

## Manual adjudication list

| sample_id | test_case | GPT-5.2 | Opus 4.6 | Manual final | Notes |
|---:|---|---|---|---|---|
| 002 | BPG5 | no_match | match | match | Candidate 3 can match after rational-function common-denominator rewriting. |
| 038 | BPG1 | no_match | match | match | Candidate 28 has `P*exp(ct)` and `P/(P+c)` terms; common-denominator rewriting matches GT. |
| 044 | BPG23 | match | no_match | match | Candidate 12 can represent the polynomial plus `P*exp(cP)` structure. |
| 080 | PO20 | no_match | match | no_match | Requires `log(Abs(v)+1)` and `x*Abs(v)^(1/3)`; candidates only provide incompatible forms such as `v**c*x`. |
| 082 | PO34 | no_match | match | no_match | Same PO-domain issue: `v**c*x` is not equivalent to `Abs(v)^(1/3)*x` over negative velocities. |
| 112 | CRK1 | match | no_match | match | Candidate 54 can reduce to polynomial terms plus a phase-shifted log-trig term. |
| 145 | MatSci1 | match | no_match | no_match | No candidate contains the required Arrhenius term `exp(-const/T)`. |
| 169 | CRK29 | no_match | match | match | Candidate 26's denominator form can represent the rational term and the linear/exponential terms. |
| 173 | MatSci14 | no_match | match | match | Candidate 68 can represent the required linear-in-`T` and `epsilon^p` cross terms under the nonnegative physical-domain rule. |
| 231 | CRK19 | match | no_match | no_match | Requires `sin(log(A+1))`; candidates only contain no-phase `cos(log(...))` or wrong log arguments. |
| 257 | CRK22 | no_match | match | match | Candidate 1 matches by common-denominator rewriting. |
| 294 | MatSci12 | verify_error | verify_error | unscorable | No final formulas/code snippets could be extracted from the log. |
| 304 | BPG3 | no_match | match | match | Candidate 100 has the needed `P^3`, `P^2`, and `P*exp(cP)` basis functions. |
| 315 | MatSci8 | match | no_match | match | Candidate 58 can represent `T*epsilon^p`, `T*epsilon`, `epsilon^p`, and `epsilon`. |
| 362 | CRK3 | no_match | match | match | Candidate 302 has `A*exp(ct)`, `A^2`, and a phase-shifted `sin(log(A+1))` term. |
| 363 | MatSci2 | verify_error | no_match | no_match | Candidates lack the required combination of `epsilon^3`, `T*exp(-epsilon)`, and `exp(-epsilon)`. |
| 368 | MatSci18 | verify_error | match | match | Candidate 653 has `epsilon^2 + epsilon*(T+c)^2`. |
| 371 | PO12 | no_match | match | no_match | `sqrt(v**c)` is not a valid substitute for `Abs(v)^(1/3)` on the PO negative-velocity domain. |
| 374 | CRK22 | no_match | match | match | Same structure as CRK22 above; common-denominator rewriting matches GT. |
| 383 | MatSci28 | no_match | match | match | Candidate 694 contains the required `T*epsilon`, `T*log(epsilon+1)`, `epsilon^2`, and related basis terms. |
| 386 | PO22 | no_match | match | match | Candidate 420 uses `exp(c/sqrt(x**(-2)))`, which can become `exp(-Abs(x))`. |
| 391 | BPG11 | no_match | match | match | Candidate 44 can represent `P`, `P^2`, `P^(1/3)`, and `P*exp(ct)`. |
| 415 | BPG16 | no_match | match | match | Candidate 28 can reduce to `P^3 + P^2 + P^(1/3)`. |
| 439 | BPG1 | no_match | match | match | Candidate 38 can match with `P*exp(ct)` plus a reciprocal rational term. |

## Takeaway

The two-judge protocol required manual intervention for only 24 of 500 cases
(4.80%). After manual adjudication, Opus 4.6 was correct on 493 of the 499
scorable cases, while GPT-5.2 was correct on 482 of the 499 scorable cases. The
main GPT-5.2 error mode was rejecting valid rational-function equivalences that
become clear after common-denominator rewriting. The main Opus 4.6 error mode
was over-relaxing domain constraints, especially treating `v**c` as if it could
stand for `Abs(v)**c` in PO tasks where negative velocities are allowed.
