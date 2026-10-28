# Code and aggregate data for wage employment exit prediction

Reproducibility material accompanying the manuscript “A Time-Aware Evaluation Protocol for Imbalanced Longitudinal Panel Prediction in Social Research: Evidence from Wage-Employment Exit”, submitted to Quality & Quantity. This repository contains the scientific content of Online Resource 2.

This resource contains Python analysis code, aggregate results, configuration and source manifests, and five figures. Online Resource 1 provides the supplementary tables and figures. Obtain the respondent-level inputs separately from the data provider; fitted models and person-level predictions are generated locally.

## Inputs

Obtain the KLIPS English-version public-use release from the Korea Labor Institute under its conditions. The 27 person and 27 household XLSX files supply model inputs. The declared audit covers 83 files, including questionnaires, codebooks and auxiliary files listed in `config/raw_file_audit.csv`. Place those files in `raw/`. The source folder lists 84 files; the unused `eklips27w.xlsx` work-history workbook was unavailable for reading and is not included in the audited manifest. It supplies no model input. Identifiers and future outcomes are excluded from predictors.

The source manifest records filenames, byte counts and SHA-256 hashes, without respondent records. The runner verifies every declared source against that manifest and rejects unlisted files that would otherwise enter the source audit. Reproducing a different source release requires an explicitly revised manifest and a fresh analysis; matching filenames alone is insufficient.

## Reproduction

Use Python 3.12 and install `requirements.txt` in an isolated environment. The recorded execution versions are in `results/execution_environment.json`. From this package directory run:

```bash
python -m pip install -r requirements.txt
python python/run_pipeline.py --analysis-only
```

Start with an empty `private/` directory for a fresh run. The cache signature checks analysis code and all declared source hashes before reusing fitted objects. Changes to either require moving the existing cache aside and starting a fresh run. The runner executes data reconstruction, fitting, sensitivity analyses, inference, figures and numerical validation in sequence. Raw inputs and regenerated `private/` files remain subject to the data provider’s conditions. The public runner does not require manuscript files or correspondence.

To redraw the five figures from the supplied aggregate outputs, run `python python/run_pipeline.py --figures-only`. report_tables.py and its extensions map aggregate sources to the 54 manuscript and supplementary tables. results/numerical_source_map.csv records those mappings. The article and Online Resource 1 explain the columns and interpretation.

## Estimation and verification

The benchmark uses unweighted person-wave fitting, validation-only probability calibration and threshold estimation, strictly forward out-of-fold meta-training, and a fixed chronological test window. The primary 1,000-draw person bootstrap conditions on fitted models, maps, thresholds and the observed historical waves; it does not estimate uncertainty from future shared temporal shocks. Sensitivity intervals and ten-seed refits address distinct sources of variation. Post hoc analyses remain exploratory. Outcome-observed selection, release-time availability, survey design and transportability delimit the interpretation.

`results/validation_report.json` and `results/extension_validation_report.json` record the primary and sensitivity checks. `results/review_validation_report.json` records 71 numerical and source checks performed by `validate_review.py`; these concern scientific outputs and contain no reviewer correspondence. `FILE_MANIFEST.csv` and `SHA256SUMS.txt` identify the distributed files and their hashes.

The 2026-10-28 verification combines archive and public-content checks, fresh source reconstruction and primary refits, and aggregate arithmetic. Source validation follows the declared manifest rather than a hard-coded file count. The fresh-execution scope is documented below; other retained experiments keep their original provenance.

## Public contents

`python/` contains analysis and validation code; `config/` records the protocol and source manifest; `results/` contains aggregate tables, summary checks and aggregate bootstrap draws; `figures/` contains five figures in PDF and PNG formats. Bootstrap rows are resampled statistics, not respondent records. Raw data, fitted objects, respondent-level predictions, manuscript source files, author declarations and editorial correspondence are excluded. The repository ignore rules also exclude local data, caches, logs and credentials.

## Independent execution verification — 28 October 2026

Fresh reads of all 83 declared source files reconstructed the 417,823-row panel, 160,102-row risk table and 139,566-row analytical table exactly, including stored-data hashes. All 54 required person/household inputs were covered; the 84th listed but unused work-history workbook could not be retrieved and was not content-audited.

Twenty fresh base-model fits (four full-training and sixteen forward-OOF fits), one unweighted meta-learner, and validation-only recalibration reproduced 54 probability-vector comparisons, 540 metric comparisons and 15 selected thresholds within the prespecified absolute tolerance 1e-10. The largest probability difference was 6.18e-12 after recalibration; the largest metric difference was 9.77e-14. Original sensitivity refits and bootstrap sampling were not all rerun in this verification. See verification_20261028/primary_refit_verification.json and the reconstruction/aggregate reports for the exact scope and checksums.

Verify the released aggregate arithmetic without raw data or third-party packages:

```bash
python python/verify_aggregate_release.py
```

This checks 120 primary intervals, 56 paired contrasts, 50 ten-seed summaries, 18 utility scenarios and 3 drift identities. `python/reproduce_primary.py --help` describes the optional fresh-refit comparison; it requires locally held respondent-level reference predictions, which are not redistributed here.
