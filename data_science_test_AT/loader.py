import pandas as pd
import numpy as np


def load_request_df(path_request: str) -> pd.DataFrame:
    request_df = pd.read_csv(
        filepath_or_buffer=path_request,
        sep=",",
        low_memory=False,
        error_bad_lines=False,
    )
    return request_df


def load_individuals_df(path_individuals: str) -> pd.DataFrame:
    individuals_df = pd.read_csv(
        filepath_or_buffer=path_individuals,
        sep=",",
        low_memory=False,
        error_bad_lines=False,
    )
    return individuals_df


def load_generic_df(path_data: str) -> pd.DataFrame:
    generic_df = pd.read_csv(
        filepath_or_buffer=path_data, low_memory=False, error_bad_lines=False
    )
    return generic_df


def load_generic_no_index_df(path_data: str) -> pd.DataFrame:
    generic_df = pd.read_csv(
        filepath_or_buffer=path_data,
        low_memory=False,
        error_bad_lines=False,
        index_col=0,
    )
    return generic_df
