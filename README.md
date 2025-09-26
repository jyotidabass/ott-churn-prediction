## OTT Churn Prediction Pipeline

End-to-end pipeline to generate synthetic streaming data, engineer behavioral features from viewing histories, train multiple churn models (XGBoost, MLP, Transformer), and produce interpretable “plausible reasons” for churn.

### Tech
- Python, Pandas, NumPy, SQLite (via `sqlite3`)
- Models: XGBoost, PyTorch MLP, PyTorch Transformer
- Interpretability: SHAP (XGBoost), permutation importance (MLP), token-level importance proxy (Transformer)
- Dockerized for easy execution

### Quickstart

Run locally:
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
python ott_churn_pipeline.py --n_users 5000 --n_titles 1200
```

Or with Docker:
```bash
docker build -t ott-churn-pipeline .
docker run --rm -it -v "%cd%/artifacts:/app/artifacts" ott-churn-pipeline
```

Artifacts will be written to `artifacts/`:
- `metrics_summary.json`: AUC/accuracy/precision/recall/F1 for all models
- `xgb_report.txt`, `mlp_report.txt`, `transformer_report.txt`
- `xgb_feature_importance.json`
- `mlp_permutation_importance.json`
- `transformer_token_importance.json`: top recent tokens (genre, device, hour) per user
- `plausible_reasons.json`: per-user churn probabilities and top plausible reasons

### What’s included
- Synthetic dataset creation for users, content, views, subscriptions, and churn labels
- Feature engineering from viewing history:
  - Recency, session gaps, binge index, device mix, late-night rate
  - Genre diversity (entropy), pause/rewatch rates, activity breadth
- Models:
  - XGBoost (tabular)
  - MLP (tabular, PyTorch)
  - Transformer (sequence + tabular fusion, PyTorch)
- Interpretability:
  - SHAP global feature importance (XGB)
  - Permutation importance (MLP)
  - Token importance proxy via encoded token magnitudes (Transformer)
  - Consolidation into human-readable “plausible reasons”

### Notes
- Defaults train on synthetic data (~5k users). Adjust `--n_users` to scale up/down.
- GPU usage is auto-detected; pass `--device cuda` to force GPU if available.

### Commit hygiene guideline
Only commit main production code (`ott_churn_pipeline.py`), `requirements.txt`, `Dockerfile`, and `README.md`. Avoid committing miscellaneous notebooks, scratch tests, or unrelated files.
