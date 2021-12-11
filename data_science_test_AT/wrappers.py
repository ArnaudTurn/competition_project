from preprocess import preprocess_pipes_from_files
from train import train_test_pipes_from_files
import argparse


def preprocess_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    preprocess_pipes_from_files(
        args.in_request_data_path, args.individual_df_path, args.output_directory
    )


def train_test_pipes_argparse_wrapper(args: argparse.Namespace) -> None:
    train_test_pipes_from_files(
        args.input_df_path, args.input_test_df_path, args.output_directory
    )
