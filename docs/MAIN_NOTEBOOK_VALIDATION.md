# MAIN_NoteBook validation

Validation date: October 6, 2026.

## Scope and execution

[MAIN_NoteBook.ipynb](../notebooks/MAIN_NoteBook.ipynb) implements the agreed minimal study in English: 33 admissible inputs, one histogram gradient boosting classifier and a constant baseline. Historical settings are reused, with no new algorithm search.

The complete 7,728,394-row CSV was read again by the repository preparation module. Sample sizes, target prevalence, event duplicates and weather availability were audited. The notebook was executed from top to bottom in a real local Jupyter kernel. All 16 Python cells completed without error and generated four figures. The notebook is below the 10 MB submission limit.

## Results reproduced

| Period | Model | Events | AP | ROC-AUC | Precision, top 10% | Recall, top 10% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| validation | prior | 50,000 | 0.3025 | 0.5000 | 30.88% | 10.21% |
| validation | hgb_full | 50,000 | 0.6969 | 0.8496 | 79.86% | 26.40% |
| test | prior | 99,999 | 0.3427 | 0.5000 | 33.50% | 9.77% |
| test | hgb_full | 99,999 | 0.7031 | 0.8362 | 80.13% | 23.38% |
| stress | prior | 50,000 | 0.3470 | 0.5000 | 35.34% | 10.19% |
| stress | hgb_full | 50,000 | 0.7092 | 0.8351 | 81.00% | 23.35% |

The six reproduced rows match the historical validation and held-out tables for sample size, prevalence, AP, ROC-AUC, top-10% recall and precision. The maximum absolute difference in the latest comparison export is 1.11e-16, below the required tolerance of 1e-9.

## Checks passed

- 21 unit tests: the original eight, ten checks for ranking metrics, missing values, transformation mappings, known versus unseen categories and documentation numbers, and three cache regression tests.
- 27 in-notebook assertions/checks: admissible columns, training-only preparation, temporal sampling, weather reason counts, change-ledger denominators, exact historical feature reproduction and metric reproduction.
- Dataset target counts and percentages in DOCUMENTATION_CIBLE.md agree with the full-file audit.
- Annual source prevalences and main-model metrics in README.md agree with the recorded tables.
- The newly computed 2021 and 2022 AP values agree with README.md and RAPPORT.md.
- No essential records or physically invalid predictor values were found in the selected samples, so the additional validity checks did not change the historical results.
- Notebook schema, code compilation, cell identifiers and local document links are valid.

## Before/after audit outputs

Full CSV exports are generated in `data/main_notebook/reports/`, including `transformation_summary.csv`, `before_after_mappings.csv`, `category_frequencies.csv`, `affected_event_summary.csv`, `record_cleaning.csv`, `invalid_numeric_values.csv`, `invalid_numeric_mappings.csv`, `weather_availability.csv` and `missingness_before_after.csv`.

A step can modify a cell more than once. The sum of step changes is distinguished from the number of events affected at least once. Type changes are reported separately. Weather-category reports distinguish retained, known-grouped, unseen and missing values by year, with category counts, positive rates and descriptive subgroup scores. They do not update the historical vocabulary.

## Reproduction commands

Use an environment containing `requirements-main-notebook.txt`:

```text
python -m unittest discover -s tests -t . -v
python scripts/validate_main_notebook.py --execute
```

Latest verified environment: Python 3.10.11, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2 and matplotlib 3.10.8. The automated validation completed all 16 cells in 244.26 seconds, including rebuilding the sample caches, and saved a 332,357-byte notebook with execution counts from 1 to 16, no error outputs and no stderr outputs. A subsequent manual rerun preserved the six score rows and all 27 consistency checks; its model training took 5.53 seconds, compared with 4.92 seconds during automated validation. The manually saved notebook is 324,056 bytes, with all 16 code cells executed and no error outputs. Execution counts can change with later manual reruns.

The previous caches were written with pandas 3.0.1 and could not be loaded by pandas 2.3.3, causing a StringDtype NotImplementedError and subsequent missing-variable errors. The cache signature now includes Python, pandas and NumPy versions. Legacy caches or caches from a different environment are rebuilt automatically from the original CSV. Generated caches are reused only after checking these versions, the CSV signature, preparation code hash, sampling settings and cache file hashes. Regression tests verify reuse in a matching environment, rebuilding after environment changes, and rebuilding a corrupt sample before loading it.

## Limits and remaining submission work

Internal consistency does not independently verify label correctness or feature availability at first report. 2021 and 2022 are historical periods already consulted. Probability calibration, uncertainty intervals and alternative weather category caps were not evaluated in this minimal notebook. The 2022 Source2 period ends September 8.

Add all group member names before submission. Include the repository Python modules and requirements, exclude data and runtime dependencies. The broader historical report contains additional models and diagnostics; this notebook deliberately reproduces only the agreed main model and baseline. README, the report, the reproduction guide and the weather-category documentation now describe the verified reproduction. The historical notebook remains available separately.
