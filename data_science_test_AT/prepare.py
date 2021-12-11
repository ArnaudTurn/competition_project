import pandas as pd
import numpy as np
import re
import joblib
from loader import load_generic_df


def select_dataset_scope(
    df: pd.DataFrame, var_to_select: list, scope: str = None
) -> pd.DataFrame:
    if scope is not None:
        return df[var_to_select].query(scope).reset_index(drop=True).copy()
    else:
        return df[var_to_select].copy()


def build_train_test_by_modality(
    train_df, test_df: pd.DataFrame, target_var: str, var: list, index_i: int = 0
) -> set:
    X_train, X_test, Y_train, Y_test = (
        pd.get_dummies(train_df[var]),
        pd.get_dummies(test_df[var]),
        train_df[target_var].values == index_i,
        test_df[target_var].values == index_i,
    )
    common_var = list(set(X_train.columns) & set(X_test.columns))
    new_var_name = [re.sub("[^a-zA-Z0-9 \n\.]", "_", i) for i in common_var]
    X_train, X_test = X_train[common_var].copy(), X_test[common_var].copy()
    X_train.columns, X_test.columns = new_var_name, new_var_name
    return X_train, X_test, Y_train, Y_test


def build_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_var: str, var: list
) -> set:
    if test_df is not None:
        X_train, X_test, Y_train, Y_test = (
            pd.get_dummies(train_df[var]),
            pd.get_dummies(test_df[var]),
            train_df[target_var].values,
            test_df[target_var].values,
        )
        common_var = list(set(X_train.columns) & set(X_test.columns))
        new_var_name = [re.sub("[^a-zA-Z0-9 \n\.]", "_", i) for i in common_var]
        X_train, X_test = X_train[common_var].copy(), X_test[common_var].copy()
        X_train.columns, X_test.columns = new_var_name, new_var_name
        return X_train, X_test, Y_train, Y_test
    elif train_df is None:
        X_test = pd.get_dummies(test_df[var])
        common_var = X_test.columns.tolist()
        new_var_name = [re.sub("[^a-zA-Z0-9 \n\.]", "_", i) for i in common_var]
        X_test.columns = new_var_name
        return None, X_test, None, None
    else:
        X_train, Y_train = pd.get_dummies(train_df[var]), train_df[target_var].values
        common_var = X_train.columns.tolist()
        new_var_name = [re.sub("[^a-zA-Z0-9 \n\.]", "_", i) for i in common_var]
        X_train.columns = new_var_name
        return X_train, None, Y_train, None


def build_train_test_pipeline(
    input_train_path: str,
    input_test_path: str,
    target_var: str,
    var: list,
    output_path: list,
) -> None:
    if input_train_path is not None:
        train_df = load_generic_df(input_train_path)
    else:
        train_df = None

    if input_test_path is not None:
        test_df = load_generic_df(input_test_path)
    else:
        test_df = None

    new_table_set = build_train_test(
        train_df=train_df,
        test_df=test_df,
        target_var=target_var,
    )
    None
