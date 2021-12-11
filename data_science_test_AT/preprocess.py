import pandas as pd
import numpy as np
import argparse
from loader import load_request_df, load_individuals_df
from utils_ import check_exist, get_unique_date
from sklearn.preprocessing import MinMaxScaler
import re


def preprocess_individuals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(
        creation_date=lambda x: 2019
        - x["individual_creation_date"]
        .fillna(999999)
        .apply(lambda y: str(y)[:4])
        .astype(int),
        age=lambda x: 2019 - x["birth_year"],
        role1_=lambda x: x["individual_role"].astype(str),
        role2_=lambda x: x["individual_role_2_label"].astype(str),
        marital_st=lambda x: x["marital_status_label"].astype(str),
        pregnancy=lambda x: x["pregnancy"].astype(str),
    )
    numeric_indi_variables = df.groupby("request_id").agg(
        {
            "individual_id": "count",
            "age": ["mean", "max", "min"],
            "creation_date": ["mean", "max", "min"],
        }
    )
    numeric_indi_variables.columns = [
        i + "_" + j for i, j in numeric_indi_variables.columns.tolist()
    ]
    cat_indi_var = (
        pd.get_dummies(
            df.set_index("request_id")[["role1_", "role2_", "marital_st", "pregnancy"]]
        )
        .reset_index()
        .groupby("request_id")
        .sum()
        .reset_index()
    )
    individual_features_df = numeric_indi_variables.merge(
        cat_indi_var, on="request_id", how="left"
    )
    return individual_features_df


def build_requests_dataset_full(
    request_df: pd.DataFrame,
    individual_df: pd.DataFrame,
    select_var: list,
    var_target: str,
    common_var: str = "request_id",
) -> pd.DataFrame:
    df_temp = request_df.copy()
    df_temp = df_temp.merge(individual_df, on=common_var, how="left").set_index(
        common_var
    )
    df_temp_dummies = pd.get_dummies(df_temp[select_var]).copy()
    var_list = df_temp_dummies.columns.tolist()
    re_var_list = [re.sub("[^a-zA-Z0-9 \n\.]", "_", i) for i in var_list]
    df_temp_dummies.columns = re_var_list
    df_temp_dummies[var_target] = df_temp[var_target]
    return df_temp_dummies


def preprocess_pipes_from_files(
    request_df_path: str,
    individual_df_path: str,
    select_var: list = None,
    var_target: str = None,
    common_var: str = None,
    output_directory: str = None,
    table_name: str = None,
) -> pd.DataFrame:

    if table_name is not None:
        unique_identifier_execution = table_name + get_unique_date()
    else:
        unique_identifier_execution = "table_" + get_unique_date()

    global_path = f"{output_directory}\{unique_identifier_execution}.csv"

    request_df = load_request_df(request_df_path)
    individual_df = load_individuals_df(individual_df_path)
    individual_df_feat = preprocess_individuals(individual_df)
    final_output = build_requests_dataset_full(
        request_df=request_df,
        individual_df=individual_df_feat,
        select_var=select_var,
        var_target=var_target,
        common_var=common_var,
    )

    check_exist(output_directory)
    final_output.to_csv(global_path)
