import pandas as pd
# from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import train_test_split, KFold
import pandas as pd


def split_data(target_col: pd.Series,
               df: pd.DataFrame,
               cfg_split: dict,
               method: str,
               custom_col: str = None):
    """
    Function to split data into X_train, X_test, y_train, y_test
    based on type of split from yaml config.

    Args:
        target_col (pd.Series): target y
        df (pd.DataFrame): matrices of featured X
        cfg_split (dict): config with structure {'type': ..., 'params': {...}}
        method (str): type split ('base', 'time', 'kfold', 'custom')
        custom_col (str, optional): custom split additional column (ex, id)

    Returns:
        X_train, X_test, y_train, y_test
    """
    params = cfg_split.get("params", {})

    # checks
    # if not isinstance(df, pd.DataFrame):
    #     raise TypeError("❌ ARG df must be pandas.DataFrame")
    # if not isinstance(target_col, (pd.Series, pd.DataFrame)):
    #     raise TypeError("❌ ARG target_col must be pandas.Series or DataFrame")
    if len(df) != len(target_col):
        raise ValueError(f"❌ Shape X ({len(df)}) and y ({len(target_col)}) not same ")

    if method == "base":
        stratify_vals = target_col if params.get("stratify", False) else None
        X_train, X_test, y_train, y_test = train_test_split(
            df,
            target_col,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            stratify=stratify_vals
        )

    elif method == "time":
        date_col = params["date_column"]
        split_date = pd.to_datetime(params["split_date"])

        mask_train = df[date_col] <= split_date
        mask_test = df[date_col] > split_date

        X_train, X_test = df.loc[mask_train], df.loc[mask_test]
        y_train, y_test = target_col.loc[mask_train], target_col.loc[mask_test]

    elif method == "kfold":
        kf = KFold(
            n_splits=params.get("n_splits", 5),
            shuffle=params.get("shuffle", True),
            random_state=params.get("random_state", 42)
        )
        return list(kf.split(df, target_col))

    elif method == "custom":
        id_col = custom_col if custom_col is not None else params.get("id_column", "global_id_ogrn")
        if id_col not in df.columns:
            raise ValueError(f"❌ Column '{id_col}' was not found in  X")

        unique_ids = df[id_col].unique()
        train_ids, test_ids = train_test_split(
            unique_ids,
            test_size=params.get("test_size", 0.2),
            random_state=params.get("random_state", 42),
            shuffle=params.get("shuffle", True)
        )

        train_mask = df[id_col].isin(train_ids)
        test_mask = df[id_col].isin(test_ids)

        X_train, X_test = df.loc[train_mask], df.loc[test_mask]
        y_train, y_test = target_col.loc[train_mask], target_col.loc[test_mask]

    else:
        raise ValueError(f"❌ Type split '{method}' is not supported")

    print(f"✅ Split done: {method}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test

