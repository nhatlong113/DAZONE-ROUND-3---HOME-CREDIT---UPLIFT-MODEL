import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def qini_curve(y, w_event, score):
    y = np.asarray(y, dtype=float)
    w_event = np.asarray(w_event, dtype=int)
    order = np.argsort(-np.asarray(score, dtype=float), kind="mergesort")
    y_s, w_s = y[order], w_event[order]
    n_t = max(int(w_event.sum()), 1)
    n_c = max(len(w_event) - n_t, 1)
    q = np.cumsum(y_s * w_s) / n_t - np.cumsum(y_s * (1 - w_s)) / n_c
    x = np.arange(1, len(y) + 1) / len(y)
    return np.r_[0.0, x], np.r_[0.0, q]


def oracle_score(y, w_event):
    y = np.asarray(y, dtype=int)
    w_event = np.asarray(w_event, dtype=int)
    return w_event * y - (1 - w_event) * y


def qini_metrics(y, w_event, score):
    x, q = qini_curve(y, w_event, score)
    auc_model = float(_trapz(q, x))
    ate = float(q[-1])
    auc_random = ate * 0.5
    _, q_perfect = qini_curve(y, w_event, oracle_score(y, w_event))
    auc_perfect = float(_trapz(q_perfect, x))
    denom = auc_perfect - auc_random
    return {
        "ate": ate,
        "auc_model": auc_model,
        "auc_random": auc_random,
        "auc_perfect": auc_perfect,
        "qini_diff": auc_model - auc_random,
        "qini_norm": (auc_model - auc_random) / denom if abs(denom) > 1e-12 else np.nan,
        "qini_ratio": auc_model / auc_random if abs(auc_random) > 1e-12 else np.nan,
    }


def qini(y, w_event, score):
    return qini_metrics(y, w_event, score)["qini_norm"]


def qini_pair(y_cash, y_card, W, cash_score, card_score):
    q_cash = qini(y_cash, (W == 1).astype(int), cash_score)
    q_card = qini(y_card, (W == 0).astype(int), card_score)
    return {"qini_cash": q_cash, "qini_card": q_card, "qini_sum": q_cash + q_card}
