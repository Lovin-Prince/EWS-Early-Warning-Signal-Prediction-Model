"""Generate a synthetic (mock) monthly panel of companies for 6-month default prediction.

One row = one company at one month-end snapshot. Each company has a hidden "distress" state z_t that
evolves over time (mean-reverting, with occasional multi-month deterioration episodes). Observable
signals are noisy functions of z_t, and the monthly default hazard depends on both the level of z_t
and its recent increase - so *trends* in the signals carry information beyond their current values.

Companies that default stop reporting (absorbing state). Financial ratios are refreshed quarterly and
published with a 1-month lag, and some monthly feeds have gaps, to mimic real reporting behaviour.
"""
import numpy as np
import pandas as pd

SEED = 42
N_MONTHS = 56                     # Jan-2022 ... Aug-2026 month-end snapshots
START = "2022-01-31"
INDUSTRIES = ["Manufacturing", "Retail", "IT Services", "Logistics", "Construction", "Pharma", "FMCG"]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_companies: int = 3000, n_months: int = N_MONTHS, seed: int = SEED,
             intercept: float = -6.9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n, T = n_companies, n_months
    dates = pd.date_range(START, periods=T, freq=pd.offsets.MonthEnd())

    industry = rng.choice(INDUSTRIES, n, p=[.22, .15, .15, .12, .14, .10, .12])
    age0 = np.clip(rng.gamma(3.0, 5.0, n), 1, 60)
    m = 0.6 * rng.normal(0, 1, n)                     # structural weakness (company's long-run mean distress)

    # ---------- hidden distress process + default events ----------
    z = np.zeros((n, T))
    z_prev = m + rng.normal(0, 0.5, n)
    episode_left = np.zeros(n, dtype=int)
    alive = np.ones(n, dtype=bool)
    default_idx = np.full(n, -1)
    last_idx = np.full(n, T - 1)
    for t in range(T):
        if t == 0:
            z_t = z_prev
        else:
            start = (episode_left == 0) & (rng.random(n) < 0.012)
            episode_left = np.where(start, rng.integers(4, 11, n), episode_left)
            drift = np.where(episode_left > 0, 0.22, 0.0)
            episode_left = np.maximum(episode_left - 1, 0)
            z_t = m + 0.93 * (z_prev - m) + drift + rng.normal(0, 0.18, n)
        z[:, t] = z_t
        if t < T - 1:                                 # default happens in month t+1 (after snapshot t)
            trend = np.maximum(z_t - z[:, t - 3], 0.0) if t >= 3 else 0.0
            hazard = _sigmoid(intercept + 1.6 * z_t + 1.5 * trend)
            hit = alive & (rng.random(n) < hazard)
            default_idx[hit] = t + 1
            last_idx[hit] = t
            alive &= ~hit
        z_prev = z_t

    def noise(scale):
        return rng.normal(0, scale, (n, T))

    # ---------- monthly signals ----------
    # Company-specific baselines (e.g. a firm that always pays 40 days late) make absolute levels ambiguous:
    # a *change* against the company's own history is what signals deterioration.
    def base(scale):
        return rng.normal(0, scale, (n, 1))

    days_beyond_terms = np.clip(8 + base(4) + 9 * z + noise(6), 0, 180)
    delinquent_invoices = rng.poisson(np.exp(-0.7 + base(0.2) + 0.55 * z))
    gst_filing_delay_days = np.clip(3 + base(2) + 4 * z + noise(4), 0, 90)
    credit_utilisation_pct = np.clip(45 + base(6) + 12 * z + noise(8), 0, 100)
    adverse_media_mentions = rng.poisson(np.exp(-1.7 + base(0.2) + 0.6 * z))
    new_litigation_cases = rng.poisson(np.exp(-2.6 + base(0.2) + 0.5 * z))
    director_changes = rng.poisson(np.exp(-3.0 + base(0.2) + 0.4 * z))
    for arr, frac in [(days_beyond_terms, .03), (gst_filing_delay_days, .03), (credit_utilisation_pct, .03)]:
        arr[rng.random((n, T)) < frac] = np.nan       # feed gaps

    cs = np.cumsum(new_litigation_cases, axis=1)      # open cases = cases raised in the last 12 months
    open_litigation_cases = cs.copy()
    open_litigation_cases[:, 12:] = cs[:, 12:] - cs[:, :-12]

    # ---------- quarterly financials, published with a 1-month lag, forward-filled ----------
    fin_full = {
        "current_ratio": np.clip(1.6 + base(0.17) - 0.35 * z + noise(0.25), 0.2, None),
        "debt_to_equity": np.exp(0.7 + base(0.2) + 0.35 * z + noise(0.3)),
        "interest_coverage": np.clip(6 + base(1.0) - 1.8 * z + noise(1.5), -10, 30),
        "net_margin": 0.07 + base(0.015) - 0.03 * z + noise(0.02),
        "revenue_growth_yoy": 0.08 + base(0.025) - 0.06 * z + noise(0.07),
    }
    q_end = np.arange(2, T, 3)                        # Mar, Jun, Sep, Dec (index 0 = Jan)
    missing_report = rng.random((n, len(q_end))) < 0.08
    q_used_idx = np.where(missing_report, np.nan, q_end[None, :].astype(float))
    q_used_idx = pd.DataFrame(q_used_idx).ffill(axis=1).to_numpy()      # latest available quarter per quarter
    t_arr = np.arange(T)
    q_pos = (t_arr // 3) - 1                          # latest quarter published before month t (-1 = none yet)
    fin = {}
    for name, full in fin_full.items():
        at_q = full[:, q_end].copy()
        at_q[missing_report] = np.nan
        at_q = pd.DataFrame(at_q).ffill(axis=1).to_numpy()
        out = np.full((n, T), np.nan)
        ok = q_pos >= 0
        out[:, ok] = at_q[:, q_pos[ok]]
        fin[name] = out
    fin_staleness = np.full((n, T), np.nan)
    ok = q_pos >= 0
    fin_staleness[:, ok] = t_arr[ok][None, :] - (q_used_idx[:, q_pos[ok]] + 1)

    # ---------- flatten to a long panel (rows exist until the company defaults) ----------
    mask = np.arange(T)[None, :] <= last_idx[:, None]
    comp = np.repeat(np.arange(n), T).reshape(n, T)[mask]
    month = np.tile(t_arr, (n, 1))[mask]
    df = pd.DataFrame({
        "company_id": np.array([f"CMP{100000 + i}" for i in range(n)])[comp],
        "month_idx": month,
        "snapshot_date": dates[month],
        "industry": industry[comp],
        "company_age_years": (age0[comp] + month / 12).round(2),
        "avg_days_beyond_terms": days_beyond_terms[mask],
        "delinquent_invoices": delinquent_invoices[mask],
        "gst_filing_delay_days": gst_filing_delay_days[mask],
        "credit_utilisation_pct": credit_utilisation_pct[mask],
        "adverse_media_mentions": adverse_media_mentions[mask],
        "new_litigation_cases": new_litigation_cases[mask],
        "director_changes": director_changes[mask],
        "open_litigation_cases": open_litigation_cases[mask],
        **{k: v[mask] for k, v in fin.items()},
        "fin_staleness_months": fin_staleness[mask],
    })
    d_idx = default_idx[comp]
    df["default_month_idx"] = np.where(d_idx >= 0, d_idx, np.nan)     # ground truth - never used as a feature
    return df


def make_label(df: pd.DataFrame, horizon: int = 6, n_months: int = N_MONTHS) -> pd.Series:
    """1 if the company defaults within the next `horizon` months, 0 if not, NaN if the window is not fully observed."""
    t, d = df["month_idx"], df["default_month_idx"]
    y = ((d > t) & (d <= t + horizon)).astype(float)
    return y.where(t + horizon <= n_months - 1)


if __name__ == "__main__":
    panel = generate()
    y = make_label(panel)
    print(panel.shape, "companies:", panel.company_id.nunique(),
          "| defaults:", int(panel.groupby("company_id").default_month_idx.first().notna().sum()))
    print("6m positive rate (labelled rows):", round(float(y.dropna().mean()), 4))
