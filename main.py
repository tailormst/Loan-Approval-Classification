import os
import sys
import joblib
import pickle
import logging
from datetime import datetime
from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import randint
from sklearn.pipeline import Pipeline as SklearnPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import warnings

# imbalanced-learn
from imblearn.over_sampling import SMOTE

# target encoder
import category_encoders as ce

# Suppress LightGBM feature name warnings
warnings.filterwarnings('ignore', message='X does not have valid feature names')

# ---------- Constants ----------
RANDOM_STATE = 42
DATA_PATH = "loan_data.csv"         # ensure this file exists in working dir
OUTPUT_DIR = "models_pkl"
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "training_log.txt")

# ---------- Setup logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("=" * 80)
logging.info(f"TRAINING STARTED - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logging.info("=" * 80)

# ---------- Load dataset ----------
df = pd.read_csv(DATA_PATH)
logging.info(f"Loaded dataset: {DATA_PATH} with shape {df.shape}")

# ---------- Outlier capping function ----------
def cap_outlier(dataframe: pd.DataFrame, column: str) -> pd.DataFrame:
    q1 = dataframe[column].quantile(0.25)
    q3 = dataframe[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    dataframe[column] = dataframe[column].clip(lower=lower_bound, upper=upper_bound)
    return dataframe

# ---------- Basic cleaning + capping ----------
df = df.dropna(how="all")

expected_cols = [
    'person_age', 'person_gender', 'person_education', 'person_income',
    'person_emp_exp', 'person_home_ownership', 'loan_amnt', 'loan_intent',
    'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
    'credit_score', 'previous_loan_defaults_on_file', 'loan_status'
]
missing = [c for c in expected_cols if c not in df.columns]
if missing:
    logging.error(f"Dataset missing expected columns: {missing}")
    raise ValueError(f"Dataset missing expected columns: {missing}")

num_cols_with_outliers = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score"
]

for col in num_cols_with_outliers:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df = cap_outlier(df, col)
logging.info("Outlier capping completed.")

# OPTIONAL feature engineering (uncomment if desired)
# df["loan_to_income"] = df["loan_amnt"] / (df["person_income"] + 1e-9)

# ---------- Split features and target ----------
target_col = "loan_status"
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

# map target to integers and save mapping
target_mapping = None
if y.dtype == "object" or y.dtype.name == "category":
    y = y.astype(str)
    unique_vals = sorted(y.unique())
    mapping = {val: i for i, val in enumerate(unique_vals)}
    y = y.map(mapping)
    target_mapping = mapping
    with open(os.path.join(OUTPUT_DIR, "target_mapping.pkl"), "wb") as f:
        pickle.dump(mapping, f)
    logging.info(f"Saved target mapping: {mapping}")

logging.info(f"Target distribution (full): {Counter(y)}")

# ---------- Identify numeric vs categorical ----------
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

logging.info(f"Numeric columns: {num_cols}")
logging.info(f"Categorical columns: {cat_cols}")

# If you added engineered features earlier, ensure num_cols/cat_cols reflect them
# (If needed, recompute num_cols/cat_cols here)

# ---------- Preprocessing ----------

# numeric transformer: median impute + standard scale
numeric_transformer = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# categorical transformer: Most frequent impute + TargetEncoder (from category_encoders)
# IMPORTANT FIX: Don't specify cols parameter - ColumnTransformer already selects the columns
categorical_transformer = SklearnPipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("target_enc", ce.TargetEncoder())  # Remove cols parameter
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"  # drop any other columns
)

# ---------- Models & (reduced) hyperparameter spaces ----------
models_param_dist = {
    "LogisticRegression": (
        LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        {
            "clf__C": [0.01, 0.1, 1.0, 10.0],
            "clf__solver": ["liblinear", "lbfgs"],
            "clf__penalty": ["l2"]
        }
    ),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RANDOM_STATE),
        {
            "clf__max_depth": randint(3, 16),
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 6),
            "clf__criterion": ["gini", "entropy"]
        }
    ),
    "RandomForest": (
        RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        {
            "clf__n_estimators": randint(50, 151),
            "clf__max_depth": randint(3, 21),
            "clf__min_samples_split": randint(2, 11),
            "clf__min_samples_leaf": randint(1, 6),
            "clf__max_features": ["sqrt", "log2"]
        }
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "clf__n_neighbors": randint(3, 16),
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2]
        }
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {
            "clf__n_estimators": randint(50, 201),
            "clf__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "clf__max_depth": randint(2, 7),
            "clf__subsample": [0.6, 0.8, 1.0]
        }
    ),
    "LightGBM": (
        LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        {
            "clf__n_estimators": [50, 100, 150],
            "clf__learning_rate": [0.01, 0.05, 0.1],
            "clf__num_leaves": [15, 31, 63],
            "clf__min_child_samples": [10, 20, 40],
            "clf__subsample": [0.6, 0.8, 1.0]
        }
    )
}

# ---------- Train/test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
logging.info(f"Train target distribution: {Counter(y_train)}")

# ---------- Diagnostic quick checks ----------
# constant columns, NaN/Inf checks (just informative)
const_cols = [c for c in X_train.columns if X_train[c].nunique(dropna=False) <= 1]
if const_cols:
    logging.warning(f"Constant columns detected and will be dropped: {const_cols}")
    X_train = X_train.drop(columns=const_cols)
    X_test = X_test.drop(columns=const_cols)

nan_cols = X_train.columns[X_train.isna().any()].tolist()
if nan_cols:
    logging.warning(f"Columns with NaN in X_train: {nan_cols}")

# ---------- Tuning loop using imblearn Pipeline (preprocessor -> SMOTE -> classifier) ----------
results = {}
best_params_all = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

N_ITER = 12   # number of random parameter combinations to try per model
SCORING = "f1"  # optimize for F1 because target is imbalanced (you can switch to "roc_auc" if preferred)

for name, (estimator, param_dist) in models_param_dist.items():
    logging.info(f"\n--- Tuning {name} ---")

    # build imblearn pipeline so SMOTE is applied only to training folds inside CV
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy="auto", k_neighbors=5)
    imb_pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", smote),
        ("clf", estimator)
    ])

    rs = RandomizedSearchCV(
        estimator=imb_pipe,
        param_distributions=param_dist,
        n_iter=N_ITER,
        scoring=SCORING,
        n_jobs=-1,
        cv=cv,
        random_state=RANDOM_STATE,
        verbose=1
    )

    # fit on training data; SMOTE and TargetEncoder operate within CV folds (no leakage)
    rs.fit(X_train, y_train)

    # capture results
    best_params = rs.best_params_
    cv_score = rs.best_score_
    y_pred = rs.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    results[name] = float(test_acc)
    best_params_all[name] = best_params

    logging.info(f"{name}: Best CV ({SCORING}) = {cv_score:.4f}, Test Accuracy = {test_acc:.4f}")
    logging.info(f"{name}: Best Params = {best_params}")
    logging.info("\n" + classification_report(y_test, y_pred, zero_division=0))

    # Save the best estimator for this model (joblib)
    fname_joblib = os.path.join(OUTPUT_DIR, f"{name.lower()}_pipeline.joblib")
    joblib.dump(rs.best_estimator_, fname_joblib, compress=3)
    logging.info(f"Saved model: {fname_joblib}")

# ---------- Save aggregated results ----------
with open(os.path.join(OUTPUT_DIR, "model_accuracies.pkl"), "wb") as f:
    pickle.dump(results, f)
with open(os.path.join(OUTPUT_DIR, "model_best_params.pkl"), "wb") as f:
    pickle.dump(best_params_all, f)

# save canonical best pipeline (fast to load later)
best_name = max(results, key=results.get)
best_joblib_path = os.path.join(OUTPUT_DIR, f"{best_name.lower()}_pipeline.joblib")
best_copy_path = os.path.join(OUTPUT_DIR, "best_pipeline.joblib")
best_pipeline = joblib.load(best_joblib_path)
joblib.dump(best_pipeline, best_copy_path, compress=3)

logging.info("\n==================== SUMMARY ====================")
logging.info(f"Accuracies (test): {results}")
logging.info(f"Best model: {best_name} -> {results[best_name]:.4f}")
logging.info(f"Saved best pipeline as: {best_copy_path}")
logging.info("All outputs (models & logs) are in: %s", OUTPUT_DIR)
logging.info("=" * 80)
logging.info("TRAINING COMPLETED SUCCESSFULLY\n")