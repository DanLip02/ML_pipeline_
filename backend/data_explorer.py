import yaml
import pandas as pd
from pathlib import Path

def load_config(cfg_path: str):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict):
    data_cfg = cfg["data"]
    df = pd.read_excel(
        data_cfg["path"],
        sheet_name=data_cfg.get("sheet_name", None),
        skiprows=data_cfg.get("skiprows", None)
    )

    if data_cfg.get("dropna", False):
        df = df.dropna()

    return df

def clean_target(df: pd.DataFrame, target_col: str, maps_num: dict) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"Целевая переменная '{target_col}' не найдена")

    before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < before:
        print(f"⚠️ Удалено {before - len(df)} строк с NaN в '{target_col}'")

    df[target_col] = df[target_col].astype(str).str.strip().str.lower()
    df["target_num"] = df[target_col].map(maps_num)

    unknown_mask = df["target_num"].isna()
    if unknown_mask.any():
        print(f"⚠️ Неизвестные рейтинги: {df.loc[unknown_mask, target_col].unique()}")
        df = df.loc[~unknown_mask]

    return df

def prepare_features(df: pd.DataFrame, cfg: dict):
    #todo get from validation api
    maps_num = {'aaa': 0, 'aa+': 1,
                'aa': 1, 'aa-': 1,
                'a+': 2, 'a': 2, 'a-': 2,
                'bbb+': 3, 'bbb': 3, 'bbb-': 3,
                'bb+': 4, 'bb': 4, 'bb-': 4,
                'b+': 5, 'b': 5, 'b-': 5,
                'ccc': 6}

    features_cfg = cfg["features"]

    num_features = features_cfg.get("numeric", [])
    cat_features = features_cfg.get("categorical", [])
    target_col = cfg["data"].get("target")

    df = df[df["skip_качество данных"] == 0]
    df['report_date'] = pd.to_datetime(df['report_date'])
    df = df.sort_values(['global_id_ogrn', 'report_date'])
    df = clean_target(df, target_col, maps_num)

    X = df[num_features + cat_features].copy()
    y = df[target_col].map(maps_num) if target_col in df.columns else None

    for col in num_features:
        X[col] = X[col].fillna(0)

    for col in cat_features:
        X[col] = X[col].fillna("missing").astype(str)

    X = X.reset_index(drop=True).values

    return X, y, num_features, cat_features


# def split_data(X, y, split_cfg: dict):
#     test_size = split_cfg.get("test_size", 0.2)
#     random_state = split_cfg.get("random_state", 42)
#     stratify = y if split_cfg.get("stratify", True) else None
#
#     if y is None:
#         raise ValueError("Target variable (y) not defined in config or dataset.")
#
#     return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def apply_data(type_data: str, data: dict=None):
    cfg_data = load_config(f"data_yaml/data_{type_data}.yaml")

    if data:
        cfg_data = data
    # cfg_split = load_config("configs/train_split.yaml")

    df = load_data(cfg_data)
    print(f"✅ Uploaded {len(df)} rows and  {df.shape[1]} columns.")

    X, y, num_features, cat_features = prepare_features(df, cfg_data)


    print(f"📊 Numerical features: {num_features}")
    print(f"🏷️ Categorical features: {cat_features}")
    if y is not None:
        print(f"🎯 Target: {cfg_data['data']['target']}")

    # X_train, X_test, y_train, y_test = split_data(X, y, cfg_split["split"])
    # print(f"🔹 Train: {X_train.shape}, Test: {X_test.shape}")

    return X, y, num_features, cat_features

# if __name__ == "__main__":
#     apply_data()