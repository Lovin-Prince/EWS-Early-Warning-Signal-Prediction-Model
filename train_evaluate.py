"""Train, tune and evaluate the 6-month-ahead default early-warning model.

Run from the repo root:  python src/train_evaluate.py

Setup ("as of today, will this company default in the next 6 months?")
  * Panel of monthly company snapshots; features use only data available at the snapshot (point-in-time).
  * Time-based split with 6-month embargo gaps so no label window overlaps across train / validation / test.
  * Hyper-parameters tuned with walk-forward (expanding-window) cross-validation, PR-AUC objective.
  * Business target: RECALL >= 80% (catch at least 4 in 5 future defaulters). Tuning objective = lowest fire
    rate needed to reach 80% recall; the alert threshold is fixed on the validation period for 80% recall.
  * A second, tighter "high-priority" tier (top 10% of scores) is reported for limited review capacity.
  * Key metrics: Precision, Recall, Fire Rate (+ lift, PR-AUC, ROC-AUC).
"""
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).parent))
from generate_data import N_MONTHS, generate, make_label  # noqa: E402
from features import build_features  # noqa: E402

SEED = 42
HORIZON = 6
TARGET_RECALL = 0.80                          # business target: catch >= 80% of upcoming defaulters
CAPACITY_FIRE_RATE = 0.10                     # tighter "high-priority" tier: top 10% of scores
SPLITS = {"train": (6, 26), "val": (33, 38), "test": (45, 49)}    # month_idx ranges (6-month embargo between)
WF_CUTOFFS = [11, 14, 17]                     # walk-forward folds: train <= cutoff, gap 6m, validate next 3m


def log(*a):
    print(*a, flush=True)


def rows(month_idx, split):
    lo, hi = SPLITS[split]
    return ((month_idx >= lo) & (month_idx <= hi)).to_numpy()


def walk_forward_folds(month_idx):
    m = month_idx.to_numpy()
    return [(np.where(m <= c)[0], np.where((m >= c + HORIZON + 1) & (m <= c + HORIZON + 3))[0]) for c in WF_CUTOFFS]


def operating_metrics(y, s, thr):
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    return {"threshold": float(thr), "precision": precision, "recall": tp / max(tp + fn, 1),
            "fire_rate": float(pred.mean()), "lift": precision / y.mean(), "tp": tp, "fp": fp, "fn": fn}


def thr_for_recall(scores, y, recall=TARGET_RECALL):
    """Score threshold that captures `recall` of the positives (computed on a reference period)."""
    return float(np.quantile(scores[np.asarray(y) == 1], 1 - recall))


def fire_rate_at_recall(est, X, y):
    """Search objective (higher = better): minus the fire rate needed to reach the target recall."""
    sc = est.predict_proba(X)[:, 1]
    return -float((sc >= thr_for_recall(sc, y)).mean())


def rank_metrics(y, s):
    return {"pr_auc": average_precision_score(y, s), "roc_auc": roc_auc_score(y, s), "base_rate": float(y.mean())}


def main():
    t0 = time.time()
    for d in ("data", "models", "reports/figures"):
        Path(d).mkdir(parents=True, exist_ok=True)

    # 1. Data, point-in-time features, 6-month-ahead label --------------------------------------------
    panel = generate(seed=SEED)
    panel.to_csv("data/mock_company_panel.csv.gz", index=False)
    X, groups = build_features(panel)
    y = make_label(panel, HORIZON, N_MONTHS)
    mi = panel["month_idx"]
    log(f"Panel: {len(panel):,} snapshots | {panel.company_id.nunique():,} companies | "
        f"{X.shape[1]} features ({len(groups['snapshot'])} snapshot + {X.shape[1]-len(groups['snapshot'])} time-series)")

    idx = {k: rows(mi, k) & y.notna().to_numpy() for k in SPLITS}
    Xtr, ytr = X[idx["train"]], y[idx["train"]].astype(int)
    Xva, yva = X[idx["val"]], y[idx["val"]].astype(int)
    Xte, yte = X[idx["test"]], y[idx["test"]].astype(int)
    for k, (a, b) in SPLITS.items():
        yy = y[idx[k]]
        log(f"  {k:5s} months {a}-{b}: rows={len(yy):6,} positives={int(yy.sum()):5,} rate={yy.mean():.2%}")

    # 2. Tuning: walk-forward CV on the training period ------------------------------------------------
    folds = walk_forward_folds(mi[idx["train"]])
    cols_ts, cols_snap = groups["timeseries"], groups["snapshot"]
    base = HistGradientBoostingClassifier(categorical_features=["industry_code"], early_stopping=False,
                                          random_state=SEED)
    space = {"learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 4, 6], "max_leaf_nodes": [8, 15, 31],
             "min_samples_leaf": [20, 50, 100], "l2_regularization": [0.0, 1.0, 5.0], "max_iter": [100, 200],
             "class_weight": [None, "balanced"]}
    search = RandomizedSearchCV(base, space, n_iter=20, scoring=fire_rate_at_recall, cv=folds,
                                random_state=SEED, n_jobs=1, refit=True)
    search.fit(Xtr[cols_ts], ytr)
    best_params = search.best_params_
    log(f"Best params: {best_params} | walk-forward CV fire rate at {TARGET_RECALL:.0%} recall = "
        f"{-search.best_score_:.4f} ({time.time()-t0:.0f}s)")

    # 3. Models compared: interpretable tree baseline, snapshot-only GBDT, time-series GBDT -----------------
    models = {}
    models["Decision Tree (baseline)"] = (Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("tree", DecisionTreeClassifier(max_depth=6, min_samples_leaf=100, class_weight="balanced",
                                        random_state=SEED))]).fit(Xtr[cols_ts], ytr), cols_ts)
    snap_model = HistGradientBoostingClassifier(categorical_features=["industry_code"], early_stopping=False,
                                                random_state=SEED, **best_params).fit(Xtr[cols_snap], ytr)
    models["GBDT - snapshot only (no history)"] = (snap_model, cols_snap)
    models["GBDT - default params"] = (HistGradientBoostingClassifier(
        categorical_features=["industry_code"], early_stopping=False, random_state=SEED).fit(Xtr[cols_ts], ytr), cols_ts)
    final = search.best_estimator_
    models["GBDT - time-series features (final)"] = (final, cols_ts)

    # 4. Evaluation: thresholds fixed on validation, applied unchanged to test ----------------------------------
    rows_out, scores = [], {}
    for name, (mdl, cols) in models.items():
        sva, ste = mdl.predict_proba(Xva[cols])[:, 1], mdl.predict_proba(Xte[cols])[:, 1]
        thr_r = thr_for_recall(sva, yva.to_numpy())                      # 80%-recall alert threshold
        thr_c = float(np.quantile(sva, 1 - CAPACITY_FIRE_RATE))          # high-priority tier threshold
        om, oc = operating_metrics(yte.to_numpy(), ste, thr_r), operating_metrics(yte.to_numpy(), ste, thr_c)
        rows_out.append({"model": name, **rank_metrics(yte.to_numpy(), ste),
                         "precision": om["precision"], "recall": om["recall"], "fire_rate": om["fire_rate"],
                         "lift": om["lift"], "threshold": thr_r,
                         "tier1_precision": oc["precision"], "tier1_recall": oc["recall"],
                         "tier1_fire_rate": oc["fire_rate"]})
        scores[name] = (sva, ste, thr_r, thr_c)
        log(f"{name:38s} PR-AUC={rows_out[-1]['pr_auc']:.3f} ROC-AUC={rows_out[-1]['roc_auc']:.3f} | "
            f"@80%-recall thr: precision={om['precision']:.3f} recall={om['recall']:.3f} fire_rate={om['fire_rate']:.3f} "
            f"lift={om['lift']:.1f}x | top-10% tier: precision={oc['precision']:.3f} recall={oc['recall']:.3f}")
    comp = pd.DataFrame(rows_out).set_index("model")
    comp.round(4).to_csv("reports/metrics_by_model.csv")

    fname = "GBDT - time-series features (final)"
    sva, ste, thr, thr_hp = scores[fname]
    y_te = yte.to_numpy()

    # 4a. Precision / recall at different fire rates (rank based, test)
    fr_rows = []
    for fr in [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        m = operating_metrics(y_te, ste, np.quantile(ste, 1 - fr))
        fr_rows.append({"fire_rate": fr, "precision": m["precision"], "recall": m["recall"], "lift": m["lift"]})
    fr_tab = pd.DataFrame(fr_rows).round(4)
    fr_tab.to_csv("reports/fire_rate_table.csv", index=False)
    log("\nPrecision/Recall by fire rate (test):\n" + fr_tab.to_string(index=False))

    # 4b. Stability across test months at the fixed threshold
    te_meta = panel.loc[idx["test"], ["company_id", "month_idx", "snapshot_date", "default_month_idx"]].copy()
    te_meta["y"], te_meta["score"] = y_te, ste
    te_meta["flag"] = te_meta["score"] >= thr
    stab = []
    for mth, gdf in te_meta.groupby("month_idx"):
        mm = operating_metrics(gdf["y"].to_numpy(), gdf["score"].to_numpy(), thr)
        stab.append({"snapshot": gdf["snapshot_date"].iloc[0].strftime("%Y-%m"), "positives": int(gdf["y"].sum()),
                     "precision": mm["precision"], "recall": mm["recall"], "fire_rate": mm["fire_rate"]})
    stab = pd.DataFrame(stab).round(4)
    stab.to_csv("reports/monthly_test_stability.csv", index=False)
    log("\nMonthly stability at fixed threshold (test):\n" + stab.to_string(index=False))

    # 4c. Company-level view: how many future defaulters were flagged at least once, and how early?
    pos = te_meta[te_meta["y"] == 1]
    per_co = pos.groupby("company_id").agg(flagged=("flag", "max"), default_idx=("default_month_idx", "first"))
    first_flag = pos[pos["flag"]].groupby("company_id")["month_idx"].min()
    lead = (per_co.loc[first_flag.index, "default_idx"] - first_flag).astype(float)
    company_recall = float(per_co["flagged"].mean())
    log(f"\nCompany-level: {int(per_co['flagged'].sum())}/{len(per_co)} upcoming defaulters flagged at least once "
        f"({company_recall:.1%}); median lead time {lead.median():.0f} months")

    # 5. Explainability: permutation importance on validation (PR-AUC drop) ------------------------------------
    pi = permutation_importance(final, Xva[cols_ts], yva, scoring="average_precision", n_repeats=3,
                                random_state=SEED, n_jobs=1)
    imp = pd.Series(pi.importances_mean, index=cols_ts).sort_values(ascending=False)
    imp.rename("pr_auc_drop").to_csv("reports/feature_importance.csv")

    # 6. Figures ------------------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, (_, ste_, thr_, _) in scores.items():
        p, r, _ = precision_recall_curve(y_te, ste_)
        ax.plot(r, p, label=f"{name} (PR-AUC {average_precision_score(y_te, ste_):.3f})")
    ax.axhline(y_te.mean(), color="grey", ls="--", lw=1, label=f"Base rate {y_te.mean():.1%}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-Recall - test period")
    ax.legend(fontsize=7); fig.tight_layout(); fig.savefig("reports/figures/precision_recall_curve.png", dpi=150)
    plt.close(fig)

    grid = np.linspace(0.01, 0.70, 70)
    pr_, rc_ = [], []
    for fr in grid:
        m = operating_metrics(y_te, ste, np.quantile(ste, 1 - fr)); pr_.append(m["precision"]); rc_.append(m["recall"])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(grid * 100, pr_, label="Precision"); ax.plot(grid * 100, rc_, label="Recall")
    ax.axhline(TARGET_RECALL, color="grey", ls="--", lw=1, label="Target recall 80%")
    ax.axvline(float((ste >= thr).mean()) * 100, color="red", ls=":", lw=1, label="Operating point")
    ax.set_xlabel("Fire rate (% of companies flagged)"); ax.set_ylabel("Precision / Recall")
    ax.set_title("Precision & recall vs fire rate - test period"); ax.legend()
    fig.tight_layout(); fig.savefig("reports/figures/precision_recall_vs_fire_rate.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#1f3b73")
    ax.set_xlabel("Drop in PR-AUC when shuffled (validation)"); ax.set_title("Top 15 features (permutation importance)")
    fig.tight_layout(); fig.savefig("reports/figures/feature_importance.png", dpi=150); plt.close(fig)

    # 7. Persist model + score the latest snapshot ("today") ------------------------------------------------------
    joblib.dump({"model": final, "features": cols_ts, "threshold": thr, "threshold_high_priority": thr_hp,
                 "horizon_months": HORIZON},
                "models/ews_default_model.joblib")
    latest = (mi == mi.max()).to_numpy()
    wl = panel.loc[latest, ["company_id", "industry", "snapshot_date"]].copy()
    wl["p_default_6m"] = final.predict_proba(X.loc[latest, cols_ts])[:, 1].round(4)
    wl["flagged"] = wl["p_default_6m"] >= thr
    wl["tier"] = np.where(wl["p_default_6m"] >= thr_hp, "High priority", np.where(wl["flagged"], "Watch", "Not flagged"))
    wl["ews_red_flag_count"] = X.loc[latest, "ews_red_flag_count"].astype(int)
    wl["avg_days_beyond_terms_3m_chg"] = X.loc[latest, "avg_days_beyond_terms_chg3"].round(1)
    wl = wl.sort_values("p_default_6m", ascending=False)
    wl[wl["flagged"]].head(300).to_csv("reports/watchlist_latest_snapshot.csv", index=False)
    log(f"\nLatest snapshot {wl['snapshot_date'].iloc[0]:%Y-%m}: {int(wl['flagged'].sum())}/{len(wl)} flagged "
        f"({wl['flagged'].mean():.1%})")

    summary = {
        "panel_rows": int(len(panel)), "companies": int(panel.company_id.nunique()),
        "n_features": int(X.shape[1]), "horizon_months": HORIZON, "splits_month_idx": SPLITS,
        "positives_test": int(y_te.sum()), "rows_test": int(len(y_te)),
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, np.floating) else v)
                        for k, v in best_params.items()},
        "walk_forward_cv_fire_rate_at_target_recall": round(float(-search.best_score_), 4),
        "target_recall": TARGET_RECALL, "high_priority_fire_rate": CAPACITY_FIRE_RATE,
        "comparison": comp.round(4).reset_index().to_dict(orient="records"),
        "fire_rate_table": fr_tab.to_dict(orient="records"),
        "company_level_recall": round(company_recall, 4), "median_lead_time_months": float(lead.median()),
        "top_features": imp.head(8).round(4).to_dict(),
    }
    Path("reports/summary.json").write_text(json.dumps(summary, indent=2, default=str))
    log(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
