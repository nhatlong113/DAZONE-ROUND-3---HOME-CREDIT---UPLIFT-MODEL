import gc

import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer

from . import config as C
from .metrics import qini_pair
from .models import predict_int_rf, predict_uplift_rf_cit

PREDICTORS = {"INT_RF": predict_int_rf, "UPLIFT_RF_CIT": predict_uplift_rf_cit}
METHODS = ["lime_importance", "shap_importance", "permutation_importance"]


def score_fn(models, name, target):
    predict = PREDICTORS[models[name]["family"]]
    model = models[name][target]
    return lambda X: predict(model, np.atleast_2d(np.asarray(X, dtype=np.float32))).astype(float)


def train_qini_table(models, X, W, y_cash, y_card):
    rows, cash_all, card_all = [], [], []
    for name in models:
        cash, card = score_fn(models, name, "cash")(X), score_fn(models, name, "card")(X)
        cash_all.append(cash)
        card_all.append(card)
        rows.append({"model": name, **qini_pair(y_cash, y_card, W, cash, card)})
    cash, card = np.mean(np.column_stack(cash_all), axis=1), np.mean(np.column_stack(card_all), axis=1)
    rows.append({"model": C.ENSEMBLE_NAME, **qini_pair(y_cash, y_card, W, cash, card)})
    return pd.DataFrame(rows)


def sample_rows(X, n_rows, seed):
    idx = np.random.default_rng(seed).choice(len(X), size=min(int(n_rows), len(X)), replace=False)
    return X[idx], idx


def lime_importance(fn, X_background, X_explain, feature_names, seed):
    explainer = LimeTabularExplainer(
        training_data=np.asarray(X_background, dtype=np.float32),
        feature_names=list(feature_names),
        mode="regression",
        discretize_continuous=True,
        random_state=seed,
    )
    rows = np.random.default_rng(seed).choice(len(X_explain), size=min(C.LIME_SAMPLE_SIZE, len(X_explain)), replace=False)
    importance = np.zeros(len(feature_names), dtype=float)
    for i in rows:
        exp = explainer.explain_instance(X_explain[i].astype(np.float32), fn, num_features=len(feature_names))
        for j, weight in exp.local_exp[next(iter(exp.local_exp))]:
            importance[int(j)] += abs(float(weight))
    return importance / len(rows)


def shap_importance(fn, X_background, X_explain):
    explainer = shap.KernelExplainer(fn, np.asarray(X_background, dtype=np.float32))
    values = np.asarray(explainer.shap_values(np.asarray(X_explain, dtype=np.float32), nsamples=C.SHAP_NSAMPLES, silent=True), dtype=float)
    return np.mean(np.abs(values), axis=0)


def permutation_importance(cash_fn, card_fn, X, y_cash, y_card, W, seed):
    rng = np.random.default_rng(seed)
    base = qini_pair(y_cash, y_card, W, cash_fn(X), card_fn(X))["qini_sum"]
    importance = np.zeros(X.shape[1], dtype=float)
    for j in range(X.shape[1]):
        drops = []
        for _ in range(C.PERMUTATION_REPEATS):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            drops.append(base - qini_pair(y_cash, y_card, W, cash_fn(X_perm), card_fn(X_perm))["qini_sum"])
        importance[j] = np.mean(drops)
    return importance


def feature_importance(models, X, W, y_cash, y_card, feature_names):
    X_imp, idx = sample_rows(X, C.IMPORTANCE_SAMPLE_SIZE, C.SEED + 1000)
    X_background, _ = sample_rows(X_imp, C.SHAP_BACKGROUND_SIZE, C.SEED + 2000)
    X_shap, _ = sample_rows(X_imp, C.SHAP_SAMPLE_SIZE, C.SEED + 3000)
    per_model = {}
    for m, name in enumerate(models, 1):
        cash_fn, card_fn = score_fn(models, name, "cash"), score_fn(models, name, "card")
        detail = pd.DataFrame({
            "feature": feature_names,
            "lime_importance": (
                lime_importance(cash_fn, X_background, X_imp, feature_names, C.SEED + m * 10 + 1)
                + lime_importance(card_fn, X_background, X_imp, feature_names, C.SEED + m * 10 + 2)
            ) / 2.0,
            "shap_importance": (shap_importance(cash_fn, X_background, X_shap) + shap_importance(card_fn, X_background, X_shap)) / 2.0,
            "permutation_importance": permutation_importance(
                cash_fn, card_fn, X_imp, y_cash[idx], y_card[idx], W[idx], C.SEED + m * 10 + 3
            ),
        })
        detail["feature_importance"] = detail[METHODS].mean(axis=1)
        per_model[f"model{m}"] = detail.sort_values("feature_importance", ascending=False).reset_index(drop=True)
        print(f"model{m} ({name}) done")
        gc.collect()
    summary = (
        pd.concat(per_model.values())
        .groupby("feature", as_index=False)[METHODS + ["feature_importance"]]
        .mean()
        .sort_values("feature_importance", ascending=False)
        .reset_index(drop=True)
    )
    return summary, per_model
