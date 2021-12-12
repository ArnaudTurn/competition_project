#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# wrappers methods                                                            #
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
from sklearn.metrics import log_loss
from datetime import datetime
import os
import joblib
import yaml


def competition_scorer(y_true, y_pred):
    return log_loss(y_true, y_pred, sample_weight=10 ** y_true)


def check_exist(folder_path: str):
    if os.path.isdir(folder_path):
        None
    else:
        os.mkdir(folder_path)


def get_unique_date():
    now = datetime.now()
    now = str(now).replace("-", "").replace(":", "").replace(" ", "_").split(".")[0]
    return now


def save_model(object, filepath: str):
    joblib.dump(
        object,
        filepath,
    )


def load_yaml(yaml_path: str) -> dict:
    with open(yaml_path, "r") as stream:
        try:
            load_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return load_yaml
