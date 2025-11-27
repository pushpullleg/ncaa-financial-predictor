# NCAA Financial Trajectory Classifier — Submission Packet

## Aim — what problem we solved
Give NCAA athletic administrators a forward-looking warning system that classifies every school as **Improving**, **Stable**, or **Declining** over the next two fiscal years so they can adjust budgets before compliance or scholarship cuts become unavoidable.

## What we built
- **Submission-ready dataset:** `final/assets/data/trajectory_ml_ready_excellent.csv` (10,332 rows × 54 engineered features) documented in `evidence/dataset_overview.md`.
- **Validated models:** Temporal CV + SMOTE pipelines culminating in the tuned XGBoost artifact stored at `final/assets/models/final_trajectory_model.joblib` (full narrative in `CSCI538_Project_Report.md`).
- **Executable demo:** CLI + regression test under `final/assets/scripts/` with evidence in `evidence/deployment_testing_notes.md`.
- **Traceability:** Notebook run logs, optimization notes, and contribution breakdowns consolidated under `evidence/` while the original exploratory work now sits in `/archive` for provenance.

## Key results
| Model | Dataset | Accuracy | ROC-AUC | Macro F1 | Improving F1 |
| --- | --- | --- | --- | --- | --- |
| Baseline persistence | Trailing class label | 0.700 | – | 0.467 | 0.412 |
| XGBoost (advanced) | `archive/today/trajectory_ml_ready_advanced.csv` | 0.554 | 0.765 | 0.516 | 0.433 |
| **XGBoost (excellent)** | `final/assets/data/trajectory_ml_ready_excellent.csv` | **0.864** | **0.965** | **0.817** | **0.647** |

The excellent model clears the rubric’s "beat heuristic by ≥10 points" requirement and finally delivers dependable recall on the Improving class.

## Why it matters
- **Preemptive decisions:** Finance offices can see a decline two budget cycles ahead, buying time for fundraising, sport realignment, or scholarship protection.
- **Sound ML practice:** Temporal splits, leakage checks, SHAP explanations, and reproducible scripts meet the "legitimate ML" expectation.
- **Self-contained packet:** Every artifact the grader needs—data fact sheet, metrics, deployment proof, templates—lives right here; deeper notebooks remain available in `archive/` if questions arise.

## Navigation map
| Location | Contents |
| --- | --- |
| `CSCI538_Project_Report.md` / `.docx` | Narrative that mirrors the course template (aim → methods → results → impact, with TODO tags for citations/figures). |
| `assets/data/` | Final CSV feeding the models. |
| `assets/models/` | `final_trajectory_model.joblib` ready for CLI use. |
| `assets/scripts/` | `predict_trajectory.py` and `test_predict_trajectory_cli.py` for demonstrations/regression tests. |
| `evidence/` | Dataset fact sheet, baseline/excellent metrics, optimization notes, deployment proof, contributions outline, references plan, notebook logs. |
| `docs/` | Course template, grading rubric, and the original strategic-plan memo for context. |

> Next polish pass: drop IEEE citations into Section 8 of the report, paste the confusion matrices/SHAP figures referenced in `evidence/supplemental_figures.md`, and finalize Appendix A contributions.
