import os
from train_model import train_ensemble_model
import yaml
import mlflow
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


def run_main(run_cfg: dict):
    # with open("yaml_runs/run_yaml_test.yaml", "r") as f:
    #     run_cfg = yaml.safe_load(f)

    NAME = os.getenv("user_name")
    PASS = os.getenv("user_pass")
    HOST = os.getenv("host", "localhost")
    PORT = os.getenv("port", 5432)
    DB = os.getenv("data_base", "postgres")
    SCHEMA = os.getenv("schema", "public")

    TYPE_DATA = run_cfg["run"]["type_data"]
    SPLIT_TYPE = run_cfg["run"]["split_type"]
    TYPE_CLASS = run_cfg["run"]["type_class"]

    full_data_cfg = run_cfg.get("full_data", None)
    model_cfg = run_cfg.get("model", None)
    metrics = run_cfg.get("metrics", None)

    #todo check name_experiment work in process
    name_experiment = model_cfg.get("model_name", None) if model_cfg is not None else None
    type_class_model = run_cfg.get("type_class_model", None)

    mlflow.set_tracking_uri(f"postgresql+psycopg2://{NAME}:{PASS}@{HOST}:{PORT}/{DB}?options=-csearch_path={SCHEMA}")

    if name_experiment is not None:
        mlflow.set_experiment(name_experiment)

    if full_data_cfg is not None and model_cfg is not None:
        return train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE, data=full_data_cfg, model=model_cfg, metrics=metrics, type_class_model=type_class_model)
    else:
        return train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE)

if __name__ == "__main__":

    DEPLOY_HOST = os.getenv("deploy_host", "localhost")
    DEPLOY_PORT = os.getenv("deploy_port", 5001)
    st.title("🚀 Credit Risk ML Platform")

    uploaded = st.file_uploader("Load YAML config", type=["yaml", "yml"])

    if uploaded:
        cfg = yaml.safe_load(uploaded)

        st.subheader("📄 Loaded config yaml")
        st.json(cfg)

        if st.button("▶ Start learning"):
            with st.spinner("Model learns, wait..."):
                run_id = run_main(cfg)

            st.success("✅ Ready!")
            st.write(f"Run ID: **{run_id}**")
            st.write(f"[Open in MLflow](http://{DEPLOY_HOST}:{DEPLOY_PORT}/#/experiments/0/runs/{run_id})")