# Airline Satisfaction — Tabular Ensemble

An end-to-end tabular classification project for the Kaggle Playground Series
S6E10 competition. The task is to rank airline passengers by their probability
of being satisfied with a flight. The evaluation metric is ROC AUC.

**Best public ROC AUC: 0.96225**

## Project highlights

- Leakage-safe 5-fold out-of-fold validation on 699,635 training rows
- LightGBM, XGBoost, CatBoost, RealMLP, and TabPFN model families
- Route-based features and cross-fitted target encoding
- Rank ensembling aligned with the ROC AUC objective
- Fold-level ablation tests for feature and ensemble decisions
- Analysis of a 112-model public OOF/test prediction library
- Submission validation for shape, IDs, finite values, and hashes

## Approach

### Validation

The project uses five fixed stratified folds. Each OOF prediction is generated
by a model that did not train on that row. Preprocessing statistics are also
fit inside each training fold. For example, missing arrival delays are filled
with the median of the corresponding training split.

This setup is used for model comparison, ensemble selection, and fold-level
stability checks.

### Model layer

The first layer combines model families with different inductive biases:

| Family | Role |
|---|---|
| LightGBM / XGBoost / CatBoost | Strong nonlinear tabular baselines |
| RealMLP | Neural representation with errors different from tree models |
| TabPFN and related TFMs | Additional model diversity |
| Route and original-data models | Domain-specific structure |

No single model produced the final score. The gain came from combining models
that made different ranking errors.

### Rank ensembling

ROC AUC depends on ordering rather than probability calibration. Predictions
are converted to percentile ranks before blending:

```python
rank = rankdata(prediction, method="average") / len(prediction)
```

This makes outputs from differently calibrated models comparable and keeps the
ensemble objective close to the competition metric.

### Residual ensemble direction

A late-stage public library provided 112 OOF/test prediction pairs. The five
best full-OOF members were combined into an equal-rank consensus. Instead of
directly averaging a weaker consensus with the strongest baseline, the pipeline
extracts only its disagreement with the existing stack:

```python
new_direction = new_consensus_rank - stack247_rank
candidate = rankdata(xwc_rank + 0.05 * new_direction)
```

The 20% version improved the local stack by approximately 0.000023 OOF AUC and
was positive on all five folds. A conservative 5% transfer weight was used for
the final diversified candidate.

### Negative-result handling

All 112 model predictions were also evaluated with an outer-fold Ridge
meta-stack. It degraded every held-out fold because the inputs were highly
correlated. The Ridge candidate was rejected rather than submitted. This test
is retained in `scripts/evaluate_new112_ridge_stack.py`.

## Results

Public leaderboard values are shown as displayed by Kaggle.

| Experiment | Public ROC AUC |
|---|---:|
| Individual tree/RealMLP models | ~0.958 |
| Cross-fitted target encoding | 0.95981 |
| Route TE + RealMLP | 0.95992 |
| Early rank ensembles | 0.95998–0.95999 |
| TabPFN candidate | 0.96064 |
| TabPFN + route + target encoding | 0.96082 |
| Large OOF stack | 0.96187 |
| Public V54 baseline | 0.96209 |
| Final public ensemble | **0.96225** |

The final stage integrates a public XWC V10 prediction artifact after validating
its IDs, row count, finite values, and published SHA256. It was not trained from
scratch in this repository. The project contribution is the validation pipeline,
model experiments, OOF/rank-ensemble analysis, residual correction, and final
candidate selection.

## Repository structure

```text
notebooks/
  01_data_research.ipynb
  02_visual_eda.ipynb
  03_preprocessing_folds.ipynb
  04_catboost.ipynb
  05_lightgbm.ipynb
  06_xgboost.ipynb
  07_ensemble.ipynb
  08_real_mlp.ipynb
  09_realmlp_kaggle.ipynb
  10_flight_distance_te.ipynb
  11_route_profile.ipynb
  12_proper_target_encoding.ipynb
  13_proper_te_cv.ipynb
  14_original_data_model.ipynb
  15_tfm_blend.ipynb
  16_auxiliary_orthogonal_stack.ipynb

scripts/
  analyze_112_stack.py
  evaluate_new112_ridge_stack.py
  build_new112_submission.py
  build_xwc_new112_submission.py

artifacts/metrics/
  JSON summaries for the main experiments
```

Generated datasets, model archives, OOF arrays, test predictions, and submission
CSVs are intentionally excluded. They are reproducible or available from the
competition and the cited public datasets.

## Reproduction

Python 3.11+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Place the competition files in `data/raw/`:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/sample_submission.csv
```

Run the numbered notebooks in order through the experiment you want to
reproduce. Notebook 03 creates the fixed folds used by later notebooks.

The late-stage scripts require `112 OOF.npz` in the project root. It is
available from:

<https://www.kaggle.com/datasets/xiffly/s6e10-112-stack-inputs-20261031>

The validated XWC artifact came from:

<https://www.kaggle.com/datasets/dennis6886/xwc-s6e10-v11-submitted-artifacts>

```bash
python scripts/analyze_112_stack.py
python scripts/evaluate_new112_ridge_stack.py
python scripts/build_xwc_new112_submission.py
```

## Engineering decisions

- OOF predictions are required before a model can enter an ensemble.
- Improvements are checked per fold, not only on aggregate AUC.
- Rank blends are preferred when model calibration differs.
- Direct blends with weaker models are avoided; residual directions are tested
  instead.
- Public leaderboard scores are treated as final external checks, not as the
  primary optimization target.
- Public artifacts retain their original attribution and are not presented as
  models trained in this repository.

## Tools

Python, pandas, NumPy, SciPy, scikit-learn, LightGBM, XGBoost, CatBoost,
PyTorch, Polars, PyArrow, RealMLP/PyTabKit, and Jupyter.
