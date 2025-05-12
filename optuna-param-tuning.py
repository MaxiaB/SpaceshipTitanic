import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

# ── constants ─────────────────────────────────────────────
SEED = 2899
N_SPLITS = 5
STORAGE = "sqlite:///optuna.db"


# ── data loading & processing ─────────────────────────────
def feature_engineering(df):
    df[["CabinDeck", "CabinNum", "CabinSide"]] = df["Cabin"].str.split("/", expand=True)
    df.drop("Cabin", axis=1, inplace=True)
    df[["FirstName", "LastName"]] = df["Name"].str.split(" ", n=1, expand=True)
    df.drop("Name", axis=1, inplace=True)
    df["FamilySize"] = df.groupby("LastName")["PassengerId"].transform("count")
    df["GroupID"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupID")["PassengerId"].transform("count")
    df["IsPort"] = df["CabinSide"].map({"P": True, "S": False})
    df.drop(
        ["CabinSide", "CabinNum", "FirstName", "LastName", "GroupID", "PassengerId"],
        axis=1,
        inplace=True,
    )
    return df


def load_data():
    df = pd.read_csv("train.csv")
    df = feature_engineering(df)
    y = df["Transported"].astype(int)
    X = df.drop("Transported", axis=1)
    return X, y


X, y = load_data()


# ── shared preprocessor ───────────────────────────────────
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    num_pipe = Pipeline(
        [("imputer", KNNImputer(n_neighbors=5)), ("scale", StandardScaler())]
    )

    return ColumnTransformer(
        [
            ("cat", cat_pipe, cat_cols),
            ("num", num_pipe, num_cols),
        ],
        remainder="drop",
    )


preprocessor = build_preprocessor(X)
cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)


# ── objective functions ────────────────────────────────────
def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])

    clf = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
    )
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()


def objective_xgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 10)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    clf = XGBClassifier(
        random_state=SEED,
        n_jobs=-1,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample,
        eval_metric="logloss",
    )
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()


def objective_lgb(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 600, step=50)
    num_leaves = trial.suggest_int("num_leaves", 16, 64)
    learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.1, log=True)
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample = trial.suggest_float("colsample_bytree", 0.5, 1.0)

    clf = LGBMClassifier(
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample,
    )
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()


def objective_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_units = []
    for i in range(n_layers):
        hidden_units.append(trial.suggest_int(f"n_units_l{i}", 32, 256, step=32))
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    lr = trial.suggest_float("learning_rate_init", 1e-5, 1e-2, log=True)

    clf = MLPClassifier(
        random_state=SEED,
        hidden_layer_sizes=tuple(hidden_units),
        activation="relu",
        alpha=alpha,
        learning_rate_init=lr,
        max_iter=1000,
        tol=1e-3,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    pipe = Pipeline([("preproc", preprocessor), ("clf", clf)])
    return cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean()


# ── run studies ────────────────────────────────────────────
if __name__ == "__main__":
    studies = {
        "rf": objective_rf,
        "xgb": objective_xgb,
        "lgb": objective_lgb,
        "mlp": objective_mlp,
    }

    for name, obj in studies.items():
        study = optuna.create_study(
            study_name=name, storage=STORAGE, load_if_exists=True, direction="maximize"
        )
        study.optimize(obj, n_trials=50)
        print(f"\n=== {name} best accuracy: {study.best_value:.4f}")
        print("params:", study.best_params)
