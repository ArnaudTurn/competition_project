#!/usr/bin/env python
# -*- coding: utf-8 -*-
###############################################################################
#                                                                             #
# Prepare class                                                               #
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
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from loader import load_generic_df
import joblib

class PrepareData():
    def __init__(self, X:pd.DataFrame) -> None:
        self.X = X.copy()
        self.__X__ = X.copy()
        self.pipeline = []

    def apply_pipeline(self,\
            module:'PrepareData') -> 'PrepareData' :
            preprocess_object = PrepareData(self.__X__)
            for _dictionnary_temp_ in module.pipeline:
                    func_temp_ = list(_dictionnary_temp_.keys())[0]
                    args_temp_ = _dictionnary_temp_[func_temp_]
                    getattr(preprocess_object, func_temp_)(**args_temp_)
            self.__X__ = preprocess_object.__X__.copy()
            self.X = preprocess_object.X.copy()
            self.pipeline = preprocess_object.pipeline
            return self

    def rescale(self,var_select:list)->'PrepareData' :
        None
    
    def resize(self, var_select: list) -> 'PrepareData':
        None

    def add_feature(self,\
            varname: str,\
            init_value) -> 'PrepareData':
            self.pipeline.append({'add_feature':{'varname':varname, 'init_value':init_value}})
            self.X[varname] = init_value
            return self

if __name__=='__main__':
    df = pd.DataFrame({"A":[1,21,3]})
    ready_df = PrepareData(df)
    ready_df.add_feature(varname="ok", init_value=1)
    joblib.dump(ready_df,"righere.pkl")
    transfo = joblib.load("righere.pkl", mmap_mode=None)
    ready_df_2 = PrepareData(df)
    ready_df_2.apply_pipeline(transfo)
    print(transfo.X==ready_df_2.X)