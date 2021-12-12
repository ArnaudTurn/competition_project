#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# train methods                                                               #
# Developed using Python 3.7.4                                                #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-12                                                            #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################


import pandas as pd
import numpy as np
import re
from lightgbm import LGBMClassifier
from utils_ import competition_scorer, check_exist, get_unique_date, save_model
from loader import load_generic_df
import joblib
import argparse


def build_train_test_by_modality(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_var: str, common_var: str, index_i: int = 0
) -> set:
    common_var_list = list(set(train_df.columns) & set(test_df.columns))
    var = [i for i in common_var_list if (target_var not in i) & (common_var not in i)]
    X_train, X_test, Y_train, Y_test = (
        train_df[var],
        test_df[var],
        train_df[target_var].values == index_i,
        test_df[target_var].values == index_i,
    )

    return X_train, X_test, Y_train, Y_test


def build_train_test(
    train_df: pd.DataFrame, test_df: pd.DataFrame, target_var: str, common_var: str
) -> set:
    if test_df is not None:
        common_var_list = list(set(train_df.columns) & set(test_df.columns))
        var = [
            i for i in common_var_list if (target_var not in i) & (common_var not in i)
        ]
        X_train, X_test, Y_train, Y_test = (
            train_df[var],
            test_df[var],
            train_df[target_var].values,
            test_df[target_var].values,
        )
        return X_train, X_test, Y_train, Y_test

    else:
        var_list = train_df.columns.tolist()
        var = [i for i in var_list if (target_var not in i) & (common_var not in i)]
        X_train, Y_train = train_df[var], train_df[target_var].values
        return X_train, None, Y_train, None


def build_assess_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, y_train: np.array, y_test: np.array
) -> set:
    lgb_model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        min_split_gain=0.1,
        class_weight={0: 1, 1: 10, 2: 100, 3: 1000},
    )
    lgb_model.fit(train_df, y_train)
    if test_df is not None:
        pred_proba = lgb_model.predict_proba(test_df)
        model_score = competition_scorer(y_test, pred_proba)
        return pred_proba, model_score, lgb_model
    else:
        return None, None, lgb_model


def train_test_pipes_from_files(
    input_df_path: str,
    target_var: str,
    common_var: str,
    input_test_df_path: str = None,
    output_directory: str = None,
) -> None:
    unique_identifier_execution = "model_" + get_unique_date()
    global_path = f"{output_directory}\{unique_identifier_execution}"
    prediction_proba_path = f"{global_path}\predictions_model.csv"
    model_perf_path = f"{global_path}\model_perf.csv"
    features_importances_path = f"{global_path}\\features_importances.csv"
    model_path = f"{global_path}\model_lgbm.pkl"
    check_exist(global_path)

    data_train_df = load_generic_df(input_df_path)

    if input_test_df_path is not None:
        data_test_df = load_generic_df(input_test_df_path)
    else:
        data_test_df = None

    X_train, X_test, Y_train, Y_test = build_train_test(
        train_df=data_train_df,
        test_df=data_test_df,
        target_var=target_var,
        common_var=common_var,
    )

    pred_proba, model_score, lgb_model = build_assess_model(
        X_train, X_test, Y_train, Y_test
    )

    columns_list = X_train.columns.tolist()

    features_importances_df = pd.DataFrame(
        {
            "variables": X_train.columns.tolist(),
            "importance_variables": lgb_model.feature_importances_,
        }
    )

    if pred_proba is not None:
        pd.DataFrame(pred_proba).to_csv(prediction_proba_path)

    if model_score is not None:
        pd.DataFrame({"Log_loss": [model_score]}).to_csv(model_perf_path)

    features_importances_df.to_csv(features_importances_path)

    save_model([lgb_model, columns_list], model_path)
