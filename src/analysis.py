import numpy as np
import pandas as pd

from . import config as C

CASH_ARM = "Cash Treatment → Cash Sign"
CARD_ARM = "Card Treatment → Card Sign"


def add_buckets(df):
    df = df.copy()
    df["age_bucket"] = pd.cut(df["cli_age"], bins=[0, 25, 35, 45, 55, np.inf], labels=["<25", "25-35", "35-45", "45-55", "55+"], right=False)
    df["termination_bucket"] = pd.cut(df["cnt_terminated_loan_12m"], bins=[-1, 0, 1, 2, 3, np.inf], labels=["0", "1", "2", "3", "4+"])
    for col, name in [
        ("last_offer_limit", "offer_limit_decile"),
        ("cnt_connected_tls", "connected_tls_decile"),
        ("cnt_total_payment", "payment_decile"),
        ("cnt_month_from_last_transaction", "recency_decile"),
    ]:
        df[name] = pd.qcut(df[col], q=10, labels=[f"D{i}" for i in range(1, 11)], duplicates="drop")
    return df


def matched_conversion(df, col, order=None):
    cash = df[df[C.TREATMENT] == "cash"].groupby(col, observed=False)[C.TARGET_CASH].mean().mul(100)
    card = df[df[C.TREATMENT] == "card"].groupby(col, observed=False)[C.TARGET_CARD].mean().mul(100)
    table = pd.DataFrame({CASH_ARM: cash, CARD_ARM: card})
    table["cash_minus_card"] = table[CASH_ARM] - table[CARD_ARM]
    table["n_customers"] = df.groupby(col, observed=False).size()
    return table.reindex([x for x in order if x in table.index]) if order else table


def segment_uplift(df, col):
    g = df.groupby([col, C.TREATMENT], observed=False)[C.TARGETS].mean().unstack(C.TREATMENT)
    n = df.groupby([col, C.TREATMENT], observed=False).size().unstack(C.TREATMENT)
    return pd.DataFrame({
        "n_cash": n["cash"],
        "n_card": n["card"],
        "uplift_cash": (g[(C.TARGET_CASH, "cash")] - g[(C.TARGET_CASH, "card")]) * 100,
        "uplift_card": (g[(C.TARGET_CARD, "card")] - g[(C.TARGET_CARD, "cash")]) * 100,
    })


def interaction_strength(df, features, min_arm=200, n_bins=10):
    rows = []
    for col in features:
        s = df[col]
        if s.nunique() > n_bins:
            s = pd.qcut(s, q=n_bins, duplicates="drop")
        tmp = df[[C.TREATMENT] + C.TARGETS].assign(_bin=s.astype(str))
        seg = segment_uplift(tmp, "_bin")
        seg = seg[(seg["n_cash"] >= min_arm) & (seg["n_card"] >= min_arm)]
        weight = seg["n_cash"] + seg["n_card"]
        row = {"feature": col, "n_segments": len(seg)}
        for t in ["uplift_cash", "uplift_card"]:
            mean = np.average(seg[t], weights=weight)
            row[f"{t}_range"] = seg[t].max() - seg[t].min()
            row[f"{t}_wstd"] = np.sqrt(np.average((seg[t] - mean) ** 2, weights=weight))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["interaction_score"] = out["uplift_cash_wstd"] + out["uplift_card_wstd"]
    return out.sort_values("interaction_score", ascending=False).reset_index(drop=True)


def strong_correlations(df, cols, threshold=0.7):
    corr = df[cols].corr()
    pairs = [
        (a, b, round(float(corr.loc[a, b]), 2))
        for i, a in enumerate(cols)
        for b in cols[i + 1:]
        if abs(corr.loc[a, b]) >= threshold
    ]
    return corr, pd.DataFrame(pairs, columns=["feature_1", "feature_2", "corr"])
