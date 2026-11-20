import gc
import json
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from . import config as C
from .metrics import qini
from .models import FAMILIES


def sample_configs(space, baseline, n_iter, seed):
    rng = random.Random(seed)
    configs = [dict(baseline)]
    seen = {json.dumps(baseline, sort_keys=True, default=str)}
    keys = list(space)
    tries = 0
    while len(configs) < n_iter and tries < n_iter * 100:
        tries += 1
        cfg = {k: rng.choice(space[k]) for k in keys}
        key = json.dumps(cfg, sort_keys=True, default=str)
        if key not in seen:
            configs.append(cfg)
            seen.add(key)
    return configs


def target_folds(w_event, y, n_splits=C.N_SPLITS, seed=C.SEED):
    strata = np.asarray(w_event, dtype=int) * 2 + np.asarray(y, dtype=int)
    fold_id = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, va) in enumerate(skf.split(np.zeros(len(y)), strata), 1):
        fold_id[va] = fold
    return fold_id


def cross_validate_target(family, X, y, w_event, params, fold_id):
    fit, predict = FAMILIES[family]
    oof = np.zeros(len(X), dtype=float)
    fold_scores = []
    for fold in range(1, fold_id.max() + 1):
        tr, va = np.where(fold_id != fold)[0], np.where(fold_id == fold)[0]
        model = fit(X[tr], y[tr], w_event[tr], params, seed=C.SEED + fold)
        oof[va] = predict(model, X[va])
        fold_scores.append(qini(y[va], w_event[va], oof[va]))
        del model
        gc.collect()
    return oof, fold_scores


def random_search(family, X, W, y_cash, y_card, configs, prefix):
    w_cash, w_card = (W == 1).astype(int), (W == 0).astype(int)
    folds = {"cash": target_folds(w_cash, y_cash), "card": target_folds(w_card, y_card)}
    rows, fold_rows, oof_store = [], [], {}
    for i, cfg in enumerate(configs, 1):
        name = f"{prefix}_{i:03d}"
        try:
            cash_oof, cash_folds = cross_validate_target(family, X, y_cash, w_cash, cfg, folds["cash"])
            card_oof, card_folds = cross_validate_target(family, X, y_card, w_card, cfg, folds["card"])
        except Exception as err:
            rows.append({"model": name, "params": json.dumps(cfg, sort_keys=True), "error": repr(err)})
            continue
        q_cash, q_card = qini(y_cash, w_cash, cash_oof), qini(y_card, w_card, card_oof)
        rows.append({
            "model": name,
            "qini_cash": q_cash,
            "qini_card": q_card,
            "qini_total": q_cash + q_card,
            "params": json.dumps(cfg, sort_keys=True),
        })
        fold_rows += [
            {"model": name, "fold": f, "qini_cash": a, "qini_card": b, "qini_total": a + b}
            for f, (a, b) in enumerate(zip(cash_folds, card_folds), 1)
        ]
        oof_store[name] = (cash_oof, card_oof)
        print(f"{name} | qini_cash={q_cash:+.6f} | qini_card={q_card:+.6f} | qini_total={q_cash + q_card:+.6f}")
    results = pd.DataFrame(rows).sort_values("qini_total", ascending=False, na_position="last").reset_index(drop=True)
    return results, pd.DataFrame(fold_rows), oof_store


def fold_summary(fold_df):
    return (
        fold_df.groupby("model")
        .agg(
            mean_qini_total=("qini_total", "mean"),
            std_qini_total=("qini_total", "std"),
            min_qini_total=("qini_total", "min"),
            mean_qini_card=("qini_card", "mean"),
            min_qini_card=("qini_card", "min"),
        )
        .sort_values("mean_qini_total", ascending=False)
    )
