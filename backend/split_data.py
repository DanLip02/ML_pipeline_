import pandas as pd
from sklearn.model_selection import train_test_split, KFold

def split_data(target_col: str, df: pd.DataFrame, cfg_split: dict, custom_col=None):
    """
    Универсальная функция разбиения данных на X_train, X_test, y_train, y_test
    в зависимости от типа сплита, описанного в конфиге.

    Args:
        df (pd.DataFrame): исходный DataFrame
        cfg_split (dict): конфиг со структурой {'type': ..., 'params': {...}}
        target_col (str): имя целевой колонки
        custom_col (str, optional): колонка для кастомного сплита (например, id)

    Returns:
        X_train, X_test, y_train, y_test
    """
    method = cfg_split["type"]
    params = cfg_split.get("params", {})

    if method == "basic":
        stratify_vals = df[params["stratify_col"]] if params.get("stratify_col") else None
        train, test = train_test_split(
            df,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            stratify=stratify_vals
        )

    elif method == "time":
        date_col = params["date_column"]
        split_date = pd.to_datetime(params["split_date"])
        train = df[df[date_col] <= split_date]
        test = df[df[date_col] > split_date]

    elif method == "kfold":
        kf = KFold(
            n_splits=params.get("n_splits", 5),
            shuffle=params.get("shuffle", True),
            random_state=params.get("random_state", 42)
        )
        # return indexes array (list)
        return list(kf.split(df))

    elif method == "custom":
        id_col = custom_col if custom_col is not None else params["id_column"]
        ids = df[id_col].unique()

        train_ids, test_ids = train_test_split(
            ids,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42)
        )
        train = df[df[id_col].isin(train_ids)]
        test = df[df[id_col].isin(test_ids)]

    else:
        raise ValueError(f"Type of split: {method} does not exist")

    X_train = train.drop(columns=[target_col])
    y_train = train[target_col].copy()
    X_test = test.drop(columns=[target_col])
    y_test = test[target_col].copy()

    return X_train, X_test, y_train, y_test
