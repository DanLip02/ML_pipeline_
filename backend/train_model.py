import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import (
    VotingClassifier,
    StackingClassifier,
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from catboost import CatBoostClassifier



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

def train_ensemble_model(X, y, cfg_path="models/model_config.yaml"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    estimators = []
    for est in cfg["estimators"]:
        model = load_model_from_cfg(est["type"], est["params"])
        estimators.append((est["name"], model))

    ensemble = build_ensemble(cfg, estimators)

    with mlflow.start_run():
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)

        y_prob = getattr(ensemble, "predict_proba", lambda X: None)(X_test)
        y_prob = y_prob[:, 1] if y_prob is not None else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        mlflow.log_param("model_name", cfg["model_name"])
        mlflow.log_param("ensemble_type", cfg["ensemble_type"])
        for est in cfg["estimators"]:
            mlflow.log_params({f"{est['name']}_{k}": v for k, v in est["params"].items()})

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)
        if auc:
            mlflow.log_metric("roc_auc", auc)

        mlflow.sklearn.log_model(ensemble, cfg["model_name"])

    return ensemble
