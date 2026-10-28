# Scope of the panel prediction benchmark

The benchmark predicts next-wave exit from wage employment among baseline wage-employed person-waves with an observed, classifiable adjacent interview. Its methodological contribution is an executable dependency sequence linking survey-state adjudication, information cutoffs, fitting, validation decisions and clustered evaluation. The individual estimators and safeguards are established methods. The KLIPS application demonstrates this implementation; transfer to another panel requires independent validation.

## Outcome and population

Test events comprise 2,006 transitions to survey-defined nonemployment, 351 to self-employment and 38 to unpaid family work (`destination_by_wave.csv`). A change of employer that retains wage employment is a continuation. Absent or unresolved follow-up is unlabelled.

The nonemployment route does not separately identify unemployment, retirement or inactivity. These can be substantively different mechanisms within that route. Correcting structurally missing worker-status detail does not make the composite endpoint homogeneous. Destination-specific refits describe sensitivity to the endpoint composition; they do not identify transition mechanisms. Worst-case prevalence bounds and retained/excluded comparisons diagnose follow-up selection without identifying prediction performance for unobserved outcomes.

## Information boundaries

| Evaluation component | What is implemented | Boundary of the claim |
|---|---|---|
| Forward OOF | Train through 7/10/13/16; score 9–11/12–14/15–17/18–20 | Every latest training outcome precedes the first scoring wave |
| Primary partition | Baselines 1–20/21–23/24–26 | Completed-wave batches: the preceding stage's final outcome and the next stage's first covariates can share a wave |
| Strict-boundary sensitivity | Training through 19; calibration 21–22; test 24–26 | A full-wave gap under the survey-wave convention |
| Earlier origins | Training endpoints 14 and 17, five learners at each | Two additional historical origins, alongside primary endpoint 20; actual data-release dates are not simulated |

Five wave-relative predictors use contemporaneous batch covariates. This transductive convention and revision-driven test reuse delimit the benchmark's application. The design does not certify real-time deployment or eliminate every possible form of future-period optimism.

## Dependence and respondent overlap

The primary bootstrap resamples people with all their test rows and multiplicity; models share the draws. It holds fitted models, calibration maps and thresholds fixed. Household-component sensitivity accommodates observed co-residence links. Neither procedure quantifies complete redevelopment, full complex-survey uncertainty or unknown future shocks shared across respondents. An interval containing zero is not an equivalence result.

The main benchmark concerns repeated panel scoring. New-person and person-disjoint comparisons concern different training and evaluation populations. Their performance gaps do not isolate an effect of training on a respondent's earlier records. The same-fit hash/history diagnostic narrows the comparison but conditions on one fit, realised assignment and observed follow-up. It establishes neither memorisation nor its absence.

## Calibration and diagnostic scope

| Output or diagnostic | Interpretation |
|---|---|
| `calibration_intercept` | Calibration-in-the-large (CITL), with the logit probability coefficient fixed at one |
| `calibration_joint_intercept` | Intercept estimated jointly with the calibration slope |
| `calibration_slope` | Slope from that joint regression; its reference value is one |
| ECE10 and Brier | Distinct finite-sample summaries; additional reporting does not guarantee subgroup or future calibration |
| Survey-weighted evaluation | Positive integrated-2018 cross-sectional weights on 26,584 observed test rows; 479 rows lack eligible weights; no full design-based variance or automatic correction for missing next-wave outcomes |
| SHAP stability | Three disjoint 1,000-row samples from the same primary 300-iteration CatBoost fit; sampling stability, not stability under refitting, across periods, or causal attribution |

## Verification dates and commands

The 2026-10-28 verification freshly reconstructed the data and executed 20 base fits, the stack and calibration. Its stored report records 54 probability-vector, 540 metric and 15 threshold comparisons within absolute tolerance 1e-10.

The 2026-10-29 verification reruns aggregate arithmetic and independently checks retained calibration, weighting and history diagnostics. It does not refit predictive models or generate new bootstrap draws. Numerical outputs and estimator implementations are retained. Reporting paths are separated from fitting imports so aggregate table construction does not load CatBoost or XGBoost.

Without respondent data or third-party packages:

```bash
python python/verify_aggregate_release.py
```

With the locally held prediction files produced by the full analysis pipeline and the declared scientific dependencies:

```bash
python python/verify_retained_predictions.py --results-dir results --private-dir /path/to/private --output-dir /path/to/check-output
```

See `verification_20261028/`, `verification_20261029/`, the protocol in `config/`, and the source manifests for the evidence and remaining boundaries. No completion score or predicted editorial decision is derived from these checks.
