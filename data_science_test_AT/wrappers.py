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


from preprocess import preprocess_pipes_from_files
from train import train_test_pipes_from_files
from predict import call_model_from_files
from kpis_performance import performance_kpis_from_files
import argparse


def preprocess_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    preprocess_pipes_from_files(
        request_df_path=args.in_request_data_path,
        individual_df_path=args.individual_df_path,
        output_directory=args.output_directory,
    )


def train_test_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    train_test_pipes_from_files(
        input_df_path=args.input_df_path,
        input_test_df_path=args.input_test_df_path,
        output_directory=args.output_directory,
    )


def call_model_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    call_model_from_files(
        input_model_path=args.input_model_path,
        input_data_path=args.input_data_path,
        output_directory=args.output_directory,
    )


def performance_kpis_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    performance_kpis_from_files(
        input_data_pred_path=args.input_data_pred_path,
        input_data_assess_on=args.input_data_assess_on,
        output_directory=args.output_directory,
    )
