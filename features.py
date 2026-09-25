"""Point-in-time time-series feature engineering.

Every feature for a company at snapshot month t uses ONLY data from months <= t (lags, rolling windows,
changes and trends). The target (default in the next 6 months) is never touched here.

Returns two feature groups so the value of history can be measured:
  * "snapshot"   - today's values only (static + current levels + a red-flag count)
  * "timeseries" - snapshot features + rolling means/max/std, 3-month changes and 6-month trend slopes
"""
import warnings

import numpy as np
import pandas as pd

MONTHLY = ["avg_days_beyond_terms", "delinquent_invoices", "gst_filing_delay_days",
           "credit_utilisation_pct", "adverse_media_mentions", "new_litigation_cases", "director_changes"]
SLOW = ["current_ratio", "debt_to_equity", "interest_coverage", "net_margin",
        "revenue_growth_yoy", "open_litigation_cases"]
INDUSTRY_CODES = {n: i for i, n in enumerate(sorted(
    ["Manufacturing", "Retail", "IT Services", "Logistics", "Construction", "Pharma", "FMCG"]))}
WINDOW = 6


def _lag_matrix(g, s: pd.Series, col: str, k: int = WINDOW) -> np.ndarray:
    """Columns = [x_t, x_(t-1), ..., x_(t-k+1)] within each company."""
    return np.column_stack([s.to_numpy()] + [g[col].shift(i).to_numpy() for i in range(1, k)])


def _red_flags(days, gst, util, cur_ratio, icr, open_lit, media):
    return ((days > 30).astype(int) + (gst > 15).astype(int) + (util > 85).astype(int)
            + (cur_ratio < 1).astype(int) + (icr < 1.5).astype(int)
            + (open_lit >= 3).astype(int) + (media >= 2).astype(int))


def build_features(df: pd.DataFrame):
    df = df.sort_values(["company_id", "month_idx"]).reset_index(drop=True) if not df.index.equals(
        pd.RangeIndex(len(df))) else df
    g = df.groupby("company_id", sort=False)
    snap, ts = {}, {}

    snap["industry_code"] = df["industry"].map(INDUSTRY_CODES).astype(int)
    snap["company_age_years"] = df["company_age_years"]
    snap["fin_staleness_months"] = df["fin_staleness_months"]

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        pos = np.arange(WINDOW - 1, -1, -1)              # position of each lag column (col0 = today = 5)
        for c in MONTHLY:
            L = _lag_matrix(g, df[c], c)
            valid = ~np.isnan(L)
            snap[c] = df[c]
            ts[f"{c}_mean3"] = np.nanmean(L[:, :3], axis=1)
            ts[f"{c}_mean6"] = np.nanmean(L, axis=1)
            ts[f"{c}_std6"] = np.nanstd(L, axis=1)
            ts[f"{c}_max6"] = np.nanmax(L, axis=1)
            ts[f"{c}_chg3"] = np.nanmean(L[:, :3], axis=1) - np.nanmean(L[:, 3:], axis=1)
            # deviation of the last 3 months from the company's own expanding-window baseline (months <= t only)
            cum_sum = df[c].fillna(0).groupby(df["company_id"], sort=False).cumsum()
            cum_n = df[c].notna().astype(int).groupby(df["company_id"], sort=False).cumsum()
            ts[f"{c}_dev_baseline"] = np.nanmean(L[:, :3], axis=1) - (cum_sum / cum_n.replace(0, np.nan)).to_numpy()
            nv = valid.sum(1)                            # NaN-aware least-squares slope over the window
            pm = (valid * pos).sum(1) / nv
            xm = np.nansum(L, axis=1) / nv
            num = np.nansum((pos - pm[:, None]) * (L - xm[:, None]), axis=1)
            den = ((pos - pm[:, None]) ** 2 * valid).sum(1)
            ts[f"{c}_slope6"] = np.where(nv >= 3, num / den, np.nan)

        for c in SLOW:
            snap[c] = df[c]
            ts[f"{c}_chg3"] = df[c] - g[c].shift(3)
            ts[f"{c}_chg6"] = df[c] - g[c].shift(6)

        # red-flag count today (snapshot rule) and its recent history (time-series)
        rf = _red_flags(df["avg_days_beyond_terms"], df["gst_filing_delay_days"], df["credit_utilisation_pct"],
                        df["current_ratio"], df["interest_coverage"], df["open_litigation_cases"],
                        df["adverse_media_mentions"])
        snap["ews_red_flag_count"] = rf
        tmp = pd.Series(rf.to_numpy(), index=df.index)
        rf_hist = np.column_stack([rf.to_numpy()] + [tmp.groupby(df["company_id"], sort=False).shift(i).to_numpy()
                                                     for i in range(1, WINDOW)])
        ts["red_flags_avg6"] = np.nanmean(rf_hist, axis=1)
        Ld = _lag_matrix(g, df["avg_days_beyond_terms"], "avg_days_beyond_terms")
        ts["stress_months_6m"] = np.nansum(Ld > 30, axis=1)

    X_snap = pd.DataFrame(snap)
    X_ts = pd.DataFrame(ts)
    X = pd.concat([X_snap, X_ts], axis=1).astype(float)
    return X, {"snapshot": list(X_snap.columns), "timeseries": list(X.columns)}
