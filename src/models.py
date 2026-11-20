import numpy as np
from sklearn.ensemble import RandomForestClassifier
from causalml.inference.tree import UpliftRandomForestClassifier

from . import config as C

RF_KEYS = [
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "min_samples_split",
    "max_features",
    "criterion",
    "class_weight",
    "bootstrap",
    "max_samples",
]


def interaction_matrix(X, w_event):
    w = np.asarray(w_event, dtype=np.float32)[:, None]
    return np.c_[X, w, X * w]


def event_control_matrices(X):
    n = len(X)
    X_event = np.c_[X, np.ones((n, 1), dtype=np.float32), X]
    X_control = np.c_[X, np.zeros((n, 1), dtype=np.float32), np.zeros_like(X)]
    return X_event, X_control


def positive_sample_weight(y, multiplier=1.0):
    y = np.asarray(y, dtype=int)
    pos = max(int(y.sum()), 1)
    neg = max(len(y) - pos, 1)
    return np.where(y == 1, (neg / pos) * float(multiplier), 1.0).astype(np.float32)


def fit_int_rf(X, y, w_event, params, seed):
    params = {**C.INT_RF_BASELINE, **params}
    model = RandomForestClassifier(**{k: params[k] for k in RF_KEYS}, n_jobs=C.N_JOBS, random_state=seed)
    sample_weight = positive_sample_weight(y, params["pos_weight_multiplier"]) if params["use_sample_weight"] else None
    model.fit(interaction_matrix(X, w_event), y, sample_weight=sample_weight)
    return model


def int_rf_event_proba(model, X):
    X_event, X_control = event_control_matrices(X)
    return model.predict_proba(X_event)[:, 1], model.predict_proba(X_control)[:, 1]


def predict_int_rf(model, X):
    p_event, p_control = int_rf_event_proba(model, X)
    return p_event - p_control


def fit_uplift_rf_cit(X, y, w_event, params, seed):
    model = UpliftRandomForestClassifier(
        control_name="control",
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        min_samples_treatment=params["min_samples_treatment"],
        max_features=int(max(1, min(int(params["max_features_cap"]), X.shape[1]))),
        evaluationFunction="CIT",
        n_jobs=C.N_JOBS,
        random_state=seed,
    )
    model.fit(X, treatment=np.where(np.asarray(w_event) == 1, "treatment", "control"), y=y)
    return model


def predict_uplift_rf_cit(model, X):
    pred = np.asarray(model.predict(X), dtype=float)
    return pred if pred.ndim == 1 else pred[:, -1]


FAMILIES = {
    "INT_RF": (fit_int_rf, predict_int_rf),
    "UPLIFT_RF_CIT": (fit_uplift_rf_cit, predict_uplift_rf_cit),
}
