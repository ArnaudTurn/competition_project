#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# predict methods                                                             #
# Developed using Python 3.7.4                                                 #
#                                                                             #
# Author: Arnaud Tauveron                                                     #
# Linkedin: https://www.linkedin.com/in/arnaud-tauveron/                      #
# Date: 2021-12-12                                                            #
# Version: 1.0.1                                                              #
#                                                                             #
###############################################################################

import pandas as pd
import numpy as np
import joblib
from loader import load_generic_df
from utils_ import get_unique_date, check_exist
from lightgbm import LGBMClassifier


def hamonize_df_to_scheme(df: pd.DataFrame, var: list):
    df_temp = df.copy()
    var_df_temp = df_temp.columns.to_list()
    common_var = list(set(var) - set(var_df_temp))
    if common_var:
        for i in common_var:
            df_temp[i] = 0
    return df_temp[var]


def call_model(df: pd.DataFrame, var: list, model) -> pd.DataFrame:
    prediction_model = pd.DataFrame(model.predict_proba(df[var]))
    return prediction_model


def call_model_from_files(
    input_model_path: str, input_data_path: str, output_directory: str
) -> None:
    unique_identifier_execution = "prediction_" + get_unique_date()
    global_path = f"{output_directory}"
    prediction_path = f"{output_directory}\{unique_identifier_execution}.csv"
    model_list = joblib.load(input_model_path, mmap_mode=None)
    data_test_df = load_generic_df(input_data_path)
    data_test_df = hamonize_df_to_scheme(df=data_test_df, var=model_list[1])
    check_exist(global_path)
    prediction_df = call_model(df=data_test_df, var=model_list[1], model=model_list[0])
    prediction_df.to_csv(prediction_path)
