import gc

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .metrics import qini_pair
from .models import fit_int_rf, fit_uplift_rf_cit, int_rf_event_proba, predict_uplift_rf_cit

SEED_OFFSET = {
    ("UPLIFT_RF_CIT", "cash"): 101,
    ("UPLIFT_RF_CIT", "card"): 202,
    ("INT_RF", "cash"): 303,
    ("INT_RF", "card"): 404,
}


def base_models():
    return [("UPLIFT_RF_CIT", cfg) for cfg in C.UPLIFT_RF_CIT_CONFIGS] + [("INT_RF", cfg) for cfg in C.INT_RF_CONFIGS]


def ensemble_cv_splits(W, y_cash, y_card, n_splits=C.N_SPLITS, seed=C.SEED):
    strata = pd.Series(W).astype(str) + "_" + pd.Series(y_cash).astype(str) + "_" + pd.Series(y_card).astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(W)), strata))


def fit_predict_base(family, cfg, target, X_fit, y_fit, w_fit, X_pred):
    seed = C.SEED + SEED_OFFSET[(family, target)] + int(cfg["n_estimators"])
    if family == "UPLIFT_RF_CIT":
        model = fit_uplift_rf_cit(X_fit, y_fit, w_fit, cfg, seed)
        uplift = predict_uplift_rf_cit(model, X_pred)
        prob = np.clip(y_fit[w_fit == 0].mean() + uplift, 0.0, 1.0)
    else:
        model = fit_int_rf(X_fit, y_fit, w_fit, cfg, seed)
        p_event, p_control = int_rf_event_proba(model, X_pred)
        uplift, prob = p_event - p_control, np.clip(p_event, 0.0, 1.0)
    return model, uplift, prob


def fit_predict_ensemble(X_fit, W_fit, y_cash_fit, y_card_fit, X_pred):
    targets = {
        "cash": (y_cash_fit, (W_fit == 1).astype(int)),
        "card": (y_card_fit, (W_fit == 0).astype(int)),
    }
    out = {t: {"uplift": {}, "prob": {}} for t in targets}
    models = {}
    for family, cfg in base_models():
        models[cfg["name"]] = {"family": family}
        for target, (y, w) in targets.items():
            model, uplift, prob = fit_predict_base(family, cfg, target, X_fit, y, w, X_pred)
            models[cfg["name"]][target] = model
            out[target]["uplift"][cfg["name"]] = uplift
            out[target]["prob"][cfg["name"]] = prob
            print(f"{cfg['name']} | {target} done")
            gc.collect()
    for target in targets:
        out[target]["uplift_mean"] = np.mean(np.column_stack(list(out[target]["uplift"].values())), axis=1)
        out[target]["prob_mean"] = np.clip(np.mean(np.column_stack(list(out[target]["prob"].values())), axis=1), 0.0, 1.0)
    return out, models


def cross_validate_ensemble(X, W, y_cash, y_card):
    rows, best = [], None
    for part, (tr, va) in enumerate(ensemble_cv_splits(W, y_cash, y_card)):
        pred, _ = fit_predict_ensemble(X[tr], W[tr], y_cash[tr], y_card[tr], X[va])
        row = {"part": part, **qini_pair(y_cash[va], y_card[va], W[va], pred["cash"]["uplift_mean"], pred["card"]["uplift_mean"])}
        rows.append(row)
        print(pd.DataFrame([row]).round(6).to_string(index=False))
        if best is None or row["qini_sum"] > best["row"]["qini_sum"]:
            best = {"row": row, "valid_idx": va, "uplift_cash": pred["cash"]["uplift_mean"], "uplift_card": pred["card"]["uplift_mean"]}
    summary = pd.DataFrame(rows)
    summary.loc[len(summary)] = {"part": "mean_5_fold", **summary[["qini_cash", "qini_card", "qini_sum"]].mean().to_dict()}
    return summary, best
