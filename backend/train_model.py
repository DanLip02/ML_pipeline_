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
from catboost import CatBoostClassifier
from data_explorer import apply_data
from split_yaml import load_config
from split_data import split_data
import logging
import os



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

        df, y, num_features, cat_features = apply_data(type_data=type_data)
        csg_split = load_config.load_split_config(split_type=split_type)
        X_train, X_test, y_train, y_test = split_data(target_col=y, df=df, cfg_split=csg_split)

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

    except Exception as e:
        logging.exception(f"Ошибка во время обучения: {e}")
        raise
