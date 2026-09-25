"""Score every active company at the latest snapshot ("today") with the saved model.

Usage (repo root):  python src/predict.py data/mock_company_panel.csv.gz
The input must contain each company's monthly history (features are lags / rolling windows).
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import build_features  # noqa: E402

if __name__ == "__main__":
    panel = pd.read_csv(sys.argv[1], parse_dates=["snapshot_date"])
    art = joblib.load("models/ews_default_model.joblib")
    X, _ = build_features(panel)
    latest = (panel["month_idx"] == panel["month_idx"].max()).to_numpy()
    out = panel.loc[latest, ["company_id", "industry", "snapshot_date"]].copy()
    out["p_default_6m"] = art["model"].predict_proba(X.loc[latest, art["features"]])[:, 1].round(4)
    out["flagged"] = out["p_default_6m"] >= art["threshold"]
    out["tier"] = ["High priority" if v >= art["threshold_high_priority"] else ("Watch" if f else "Not flagged")
                   for v, f in zip(out["p_default_6m"], out["flagged"])]
    out = out.sort_values("p_default_6m", ascending=False)
    out.to_csv("reports/scored_latest_snapshot.csv", index=False)
    print(f"{int(out['flagged'].sum())} of {len(out)} companies flagged (threshold {art['threshold']:.3f})")
    print(out.head(10).to_string(index=False))
