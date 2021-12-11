import pandas as pd
import numpy as np
from data_science_test_AT.utils_ import load_yaml
from data_science_test_AT.preprocess import (
    preprocess_pipes_from_files,
)
from data_science_test_AT.train import train_test_pipes_from_files, build_train_test
from data_science_test_AT.predict import call_model_from_files
from data_science_test_AT.kpis_performance import performance_kpis_from_files
import yaml

if __name__ == "__main__":
    load_config = load_yaml(yaml_path="competition_project\config_paths.yaml")

    if load_config["config_general"]["build_preprocess"] is not None:
        preprocess_pipes_from_files(
            request_df_path=load_config["config_preprocess"]["input_data_request"],
            individual_df_path=load_config["config_preprocess"][
                "input_data_individuals"
            ],
            select_var=load_config["config_preprocess"]["variables_selected"],
            common_var=load_config["config_preprocess"]["variable_for_join"],
            var_target=load_config["config_preprocess"]["variable_target"],
            output_directory=load_config["config_preprocess"]["output_path"],
            table_name=load_config["config_preprocess"]["tablename"],
        )

    if load_config["config_general"]["build_preprocess_test"] is not None:
        preprocess_pipes_from_files(
            request_df_path=load_config["config_preprocess_test"]["input_data_request"],
            individual_df_path=load_config["config_preprocess_test"][
                "input_data_individuals"
            ],
            select_var=load_config["config_preprocess_test"]["variables_selected"],
            common_var=load_config["config_preprocess"]["variable_for_join"],
            var_target=load_config["config_preprocess"]["variable_target"],
            output_directory=load_config["config_preprocess_test"]["output_path"],
            table_name=load_config["config_preprocess_test"]["tablename"],
        )

    if load_config["config_general"]["build_train"] is not None:
        train_test_pipes_from_files(
            input_df_path=load_config["config_train"]["input_data"],
            target_var=load_config["config_preprocess"]["variable_target"],
            common_var=load_config["config_preprocess"]["variable_for_join"],
            input_test_df_path=load_config["config_train"]["input_test_data"],
            output_directory=load_config["config_train"]["output_directory"],
        )

    if load_config["config_general"]["build_predict"] is not None:
        call_model_from_files(
            input_model_path=load_config["config_predict"]["input_model"],
            input_data_path=load_config["config_predict"]["input_data"],
            output_directory=load_config["config_predict"]["output_directory"],
        )

    if load_config["config_general"]["build_kpis"] is not None:
        performance_kpis_from_files(
            input_data_pred_path=load_config["config_kpis"]["input_data_pred"],
            input_data_assess_on=load_config["config_kpis"]["input_data_assess"],
            target_var=load_config["config_preprocess"]["variable_target"],
            output_directory=load_config["config_kpis"]["output_directory"],
        )
