import yaml
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
from data_explorer import apply_data
from split_yaml import load_config
from split_data import split_data
import logging
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    filename=os.path.join(log_dir, "train_ensemble.log"),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def load_model_from_cfg(model_type, params):
    if model_type == "RandomForestClassifier":
        return RandomForestClassifier(**params)
    elif model_type == "LogisticRegression":
        return LogisticRegression(**params)
    elif model_type == "XGBClassifier":
        return XGBClassifier(**params)
    elif model_type == "CatBoostClassifier":
        return CatBoostClassifier(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def build_ensemble(cfg, estimators):
    ensemble_type = cfg.get("ensemble_type", "voting").lower()

    if ensemble_type == "voting":
        return VotingClassifier(estimators=estimators, voting=cfg.get("voting", "hard"))

    elif ensemble_type == "stacking":
        final_estimator = load_model_from_cfg(cfg["final_estimator"]["type"], cfg["final_estimator"]["params"])
        return StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            passthrough=cfg.get("passthrough", False)
        )

    elif ensemble_type == "bagging":
        base_model = load_model_from_cfg(cfg["base_estimator"]["type"], cfg["base_estimator"]["params"])
        return BaggingClassifier(
            estimator=base_model,
            n_estimators=cfg.get("n_estimators", 10),
            random_state=42
        )

    else:
        raise ValueError(f"Unsupported ensemble_type: {ensemble_type}")

def train_ensemble_model(type_data: str, type_class: str, split_type: str="base"):
    cfg_path = f"models/model_config_{type_class}.yaml"
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)

        X, y, num_features, cat_features = apply_data(type_data=type_data)
        csg_split = load_config.load_split_config(split_type=split_type)
        X_train, X_test, y_train, y_test = split_data(target_col=y, df=X, cfg_split=csg_split, method=split_type)

        X_train = pd.DataFrame(X_train, columns=num_features + cat_features)
        X_test = pd.DataFrame(X_test, columns=num_features + cat_features)

        dupes = X_train.columns[X_train.columns.duplicated()]

        if len(dupes) > 0:
            print("duplicates columns was found:", list(dupes))
            X_train = X_train.loc[:, ~X_train.columns.duplicated()].copy()
            X_test = X_test.loc[:, ~X_test.columns.duplicated()].copy()

        numeric_transformer = Pipeline(steps=[
            ("scaler", SimpleImputer(strategy="median"))
        ])

        categorical_transformer = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, num_features),
                ("cat", categorical_transformer, cat_features)
            ]
        )

        estimators = []
        for est in cfg["ensemble"]["estimators"]:
            model = load_model_from_cfg(est["type"], est["params"])
            estimators.append((est["name"], model))

        ensemble = build_ensemble(cfg, estimators)

        model_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("ensemble", ensemble)
        ])

        config_model = cfg["ensemble"]

        with mlflow.start_run():
            model_pipeline.fit(X_train, y_train)
            y_pred = model_pipeline.predict(X_test)

            y_prob = getattr(model_pipeline, "predict_proba", lambda X: None)(X_test)
            y_prob = y_prob[:, 1] if y_prob is not None else None

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro")
            auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

            mlflow.log_param("model_name", cfg["model_name"])
            mlflow.log_param("ensemble_type", config_model["type"])
            for est in config_model["estimators"]:
                mlflow.log_params({f"{est['name']}_{k}": v for k, v in est["params"].items()})

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1", f1)
            if auc:
                mlflow.log_metric("roc_auc", auc)

            input_example = X_train.head(1)
            signature = infer_signature(X_train, y_train)

            mlflow.sklearn.log_model(model_pipeline,
                                     name=cfg["model_name"],
                                     input_example=input_example,
                                     signature=signature)

        return model_pipeline

    except Exception as e:
        logging.exception(f"Ошибка во время обучения: {e}")
        raise
