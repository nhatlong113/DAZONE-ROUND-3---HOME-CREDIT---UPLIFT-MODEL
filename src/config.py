from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_RAW = RAW_DIR / "dazone_sample.csv"
TEST_RAW = RAW_DIR / "dazone_sample_test.csv"
TRAIN_CLEANED = PROCESSED_DIR / "cleaned_dazone_sample.csv"
TRAIN_FEATURE = PROCESSED_DIR / "df_feature.csv"
TRAIN_TARGET = PROCESSED_DIR / "df_target.csv"
TEST_FEATURE = PROCESSED_DIR / "test_df_feature.csv"
TRANSFORMERS = PROCESSED_DIR / "transformers.joblib"

SEED = 42
N_SPLITS = 5
N_JOBS = 4

ID_COL = "customer_id"
TREATMENT = "treatment"
TREATMENT_CASH = "treatment_cash"
TARGET_CASH = "cash_sign_30d"
TARGET_CARD = "card_sign_30d"
TARGETS = [TARGET_CASH, TARGET_CARD]

HIGH_MISSING_COLS = [
    "f_appointment_3m",
    "max_cntr_len_fi_12m",
    "ratio_monthly_amt_l6m_vs_n6m",
    "ratio_vol_hcvn_cash_appl_12m",
    "cnt_month_from_first_transaction",
    "cnt_month_from_last_transaction",
]

NUM_CONT_COLS = [
    "amt_terminated_loan_12m",
    "cli_emp_period",
    "last_offer_limit",
    "cnt_month_from_last_appl",
    "cnt_total_payment",
    "cnt_fail_tls_3m",
    "cnt_connected_tls",
    "cnt_bank_inst_appl_12m",
    "cnt_fi_inst_appl_12m",
    "min_sign_vol",
    "ratio_conn_tls_weekend_12m",
    "ratio_conn_tls_endmonth_12m",
    "avg_sign_vol",
    "cli_age",
    "cnt_month_status1",
]

NUM_CAT_COLS = [
    "f_approve_last_process",
    "worst_credit_status",
    "f_ever_default",
    "cnt_report_bank_6m",
    "cnt_report_institution_12m",
    "cnt_lender_has2pay_last_24m",
    "cnt_terminated_loan_12m",
    "cnt_month_over3_credit_card_6m",
    "cnt_noinsurance_loan_24m",
    "cnt_earlyterminated_cash",
    "cnt_earlyterminted_pos",
    "cnt_inbound_tls_12m",
]

NOMINAL_COLS = [
    "treatment",
    "cli_job",
    "cli_marital_status",
    "cli_contact_region_level",
    "last_process_channel",
    "risk_grp",
    "cli_contact_region_population_grp",
]

ORDINAL_COLS = ["cli_education"]
EDUCATION_ORDER = ["ELEMENTARY_SCHOOL", "PTCS", "PTTH", "BACHELORS", "MASTERS"]

UPLIFT_RF_CIT_BASELINE = {
    "n_estimators": 80,
    "max_depth": 5,
    "min_samples_leaf": 120,
    "min_samples_treatment": 40,
    "max_features_cap": 12,
}

UPLIFT_RF_CIT_SPACE = {
    "n_estimators": [60, 80, 120, 160, 220],
    "max_depth": [3, 4, 5, 6, 8, None],
    "min_samples_leaf": [50, 80, 100, 120, 160, 220],
    "min_samples_treatment": [20, 30, 40, 60, 80, 120],
    "max_features_cap": [6, 8, 10, 12, 16, 20, 32],
}

UPLIFT_RF_CIT_N_ITER = 30
UPLIFT_RF_CIT_SEARCH_SEED = 42

INT_RF_BASELINE = {
    "n_estimators": 320,
    "max_depth": 9,
    "min_samples_leaf": 80,
    "min_samples_split": 2,
    "max_features": "sqrt",
    "criterion": "gini",
    "class_weight": "balanced_subsample",
    "bootstrap": True,
    "max_samples": None,
    "use_sample_weight": True,
    "pos_weight_multiplier": 1.0,
}

INT_RF_SPACE = {
    "n_estimators": [180, 240, 320, 420, 560],
    "max_depth": [6, 8, 9, 10, 12, None],
    "min_samples_leaf": [30, 50, 80, 120, 160],
    "min_samples_split": [2, 20, 50, 100],
    "max_features": ["sqrt", "log2", 0.35, 0.50, 0.75],
    "criterion": ["gini", "entropy"],
    "class_weight": ["balanced_subsample", "balanced", None],
    "bootstrap": [True],
    "max_samples": [None, 0.70, 0.85],
    "use_sample_weight": [True, False],
    "pos_weight_multiplier": [0.60, 0.80, 1.00, 1.20, 1.50],
}

INT_RF_N_ITER = 40
INT_RF_SEARCH_SEED = 2026

UPLIFT_RF_CIT_CONFIGS = [
    {"name": "CFG028_UPLIFT_RF_CIT", "n_estimators": 80, "max_depth": 8, "min_samples_leaf": 120, "min_samples_treatment": 20, "max_features_cap": 32},
    {"name": "CFG009_UPLIFT_RF_CIT", "n_estimators": 120, "max_depth": 5, "min_samples_leaf": 80, "min_samples_treatment": 30, "max_features_cap": 32},
    {"name": "CFG011_UPLIFT_RF_CIT", "n_estimators": 120, "max_depth": 5, "min_samples_leaf": 160, "min_samples_treatment": 40, "max_features_cap": 32},
]

INT_RF_CONFIGS = [
    {"name": "INT_RF_RS_027", "n_estimators": 560, "max_depth": 6, "min_samples_leaf": 80, "min_samples_split": 50, "max_features": "log2", "criterion": "entropy", "class_weight": "balanced", "bootstrap": True, "max_samples": None, "use_sample_weight": False, "pos_weight_multiplier": 1.0},
    {"name": "INT_RF_RS_028", "n_estimators": 420, "max_depth": 6, "min_samples_leaf": 30, "min_samples_split": 2, "max_features": "log2", "criterion": "gini", "class_weight": "balanced", "bootstrap": True, "max_samples": None, "use_sample_weight": False, "pos_weight_multiplier": 1.0},
]

ENSEMBLE_NAME = "ENSEMBLE_3_UPLIFT_RF_CIT_2_INT_RF"

IMPORTANCE_SAMPLE_SIZE = 800
LIME_SAMPLE_SIZE = 30
SHAP_BACKGROUND_SIZE = 40
SHAP_SAMPLE_SIZE = 80
SHAP_NSAMPLES = 129
PERMUTATION_REPEATS = 3
