from train_model import train_ensemble_model
import yaml

if __name__ == "__main__":

    with open("yaml_runs/run_yaml_test.yaml", "r") as f:
        run_cfg = yaml.safe_load(f)

    TYPE_DATA = run_cfg["run"]["type_data"]
    SPLIT_TYPE = run_cfg["run"]["split_type"]
    TYPE_CLASS = run_cfg["run"]["type_class"]

    full_data_cfg = run_cfg.get("full_data", None)
    model_cfg = run_cfg.get("model", None)

    if full_data_cfg is not None and model_cfg is not None:
        train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE, data=full_data_cfg, model=model_cfg)
    else:
        train_ensemble_model(type_data=TYPE_DATA, type_class=TYPE_CLASS, split_type=SPLIT_TYPE)