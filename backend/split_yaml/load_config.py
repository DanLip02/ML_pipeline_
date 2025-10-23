import yaml
from pathlib import Path

def load_split_config(split_type: str, base_dir: str = "split_yaml"):
    """
    Загружает YAML-конфиг для выбранного типа сплита.

    Args:
        split_type (str): тип сплита ('basic', 'time', 'custom', 'kfold' и т.д.)
        base_dir (str): директория, где хранятся YAML-конфиги

    Returns:
        dict: словарь с параметрами сплита
    """
    split_path = Path(base_dir) / f"split_{split_type}.yaml"

    if not split_path.exists():
        raise FileNotFoundError(f"Config for split: '{split_type}' was not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg["split"]