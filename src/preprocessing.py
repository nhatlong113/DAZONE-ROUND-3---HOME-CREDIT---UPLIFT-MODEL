import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder

from . import config as C


def load_raw(path):
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])


def missing_summary(df):
    null_pct = df.isnull().mean() * 100
    bins = [-np.inf, 0, 5, 20, 50, 80, np.inf]
    labels = ["no", "very_few", "few", "medium", "many", "very_many"]
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.count(),
        "null": df.isnull().sum(),
        "null_pct": null_pct.round(3),
        "unique": df.nunique(dropna=True),
        "missing_level": pd.cut(null_pct, bins=bins, labels=labels),
    })


def _frame(values, columns, index):
    return pd.DataFrame(values, columns=columns, index=index)


def fit_transformers(df):
    tf = {
        "imputer": SimpleImputer(strategy="median"),
        "scaler": MinMaxScaler(),
        "num_cat_imputer": SimpleImputer(strategy="most_frequent"),
        "num_cat_scaler": MinMaxScaler(),
        "nominal_imputer": SimpleImputer(strategy="most_frequent"),
        "nominal_encoder": OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
        "ordinal_imputer": SimpleImputer(strategy="most_frequent"),
        "ordinal_encoder": OrdinalEncoder(categories=[C.EDUCATION_ORDER], dtype=int),
        "ordinal_scaler": MinMaxScaler(),
    }
    idx = df.index
    tf["scaler"].fit(tf["imputer"].fit_transform(df[C.NUM_CONT_COLS]))
    tf["num_cat_scaler"].fit(tf["num_cat_imputer"].fit_transform(df[C.NUM_CAT_COLS]))
    nominal_raw = _frame(tf["nominal_imputer"].fit_transform(df[C.NOMINAL_COLS]), C.NOMINAL_COLS, idx)
    tf["nominal_encoder"].fit(nominal_raw)
    ordinal_raw = _frame(tf["ordinal_imputer"].fit_transform(df[C.ORDINAL_COLS]), C.ORDINAL_COLS, idx)
    tf["ordinal_scaler"].fit(tf["ordinal_encoder"].fit_transform(ordinal_raw))
    return tf


def transform(df, tf):
    idx = df.index
    num_cont = tf["scaler"].transform(tf["imputer"].transform(df[C.NUM_CONT_COLS]))
    num_cat = tf["num_cat_scaler"].transform(tf["num_cat_imputer"].transform(df[C.NUM_CAT_COLS]))
    nominal_raw = _frame(tf["nominal_imputer"].transform(df[C.NOMINAL_COLS]), C.NOMINAL_COLS, idx)
    nominal = tf["nominal_encoder"].transform(nominal_raw)
    ordinal_raw = _frame(tf["ordinal_imputer"].transform(df[C.ORDINAL_COLS]), C.ORDINAL_COLS, idx)
    ordinal = tf["ordinal_scaler"].transform(tf["ordinal_encoder"].transform(ordinal_raw))
    return pd.concat([
        _frame(num_cont, C.NUM_CONT_COLS, idx),
        _frame(num_cat, C.NUM_CAT_COLS, idx),
        _frame(nominal, tf["nominal_encoder"].get_feature_names_out(C.NOMINAL_COLS), idx),
        _frame(ordinal, C.ORDINAL_COLS, idx),
    ], axis=1)


def load_model_data(feature_path=C.TRAIN_FEATURE, target_path=C.TRAIN_TARGET):
    feature = pd.read_csv(feature_path)
    target = pd.read_csv(target_path)
    feat_cols = [c for c in feature.columns if c != C.TREATMENT_CASH]
    X = feature[feat_cols].to_numpy(dtype=np.float32)
    W = feature[C.TREATMENT_CASH].astype(int).to_numpy()
    y_cash = target[C.TARGET_CASH].astype(int).to_numpy()
    y_card = target[C.TARGET_CARD].astype(int).to_numpy()
    return X, W, y_cash, y_card, feat_cols


def load_test_matrix(feat_cols, path=C.TEST_FEATURE):
    return pd.read_csv(path)[feat_cols].to_numpy(dtype=np.float32)
