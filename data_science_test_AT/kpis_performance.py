#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Kpis methods                                                                #
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
from utils_ import competition_scorer, get_unique_date
from loader import load_generic_no_index_df


def assess_score(
    df_pred: pd.DataFrame, df_real: pd.DataFrame, target_var: str
) -> pd.DataFrame:
    value_score = competition_scorer(df_real[target_var], df_pred)
    result_score = pd.DataFrame({"score": [value_score]})
    return result_score


def performance_kpis_from_files(
    input_data_pred_path: str,
    input_data_assess_on: str,
    target_var: str,
    output_directory: str,
) -> None:
    unique_identifier_execution = "kpi_" + get_unique_date()
    global_path = f"{output_directory}"
    kpis_score_path = f"{output_directory}\score_{unique_identifier_execution}.csv"
    data_pred_df = load_generic_no_index_df(input_data_pred_path)
    data_assess_df = load_generic_no_index_df(input_data_assess_on)
    results_kpis = assess_score(
        df_pred=data_pred_df, df_real=data_assess_df, target_var=target_var
    )
    results_kpis.to_csv(kpis_score_path)
