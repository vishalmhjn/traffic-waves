from pathlib import Path
import argparse

from utils import format_dates, func_path_data, setup_logging

from call_data_api import data_collector
from process_data import data_processor
from train import model_trainer
from predict import predictor

logging = setup_logging(file_name="workflow.log")

BASE_PATH_DATA = Path("../data")
PATH_PREDICTIONS = Path("../predictions")

# Define the specific paths using the base paths
raw_data_folder = BASE_PATH_DATA / "raw_data"
processed_data_folder = BASE_PATH_DATA / "processed_data"
historical_data_folder = BASE_PATH_DATA / "historical_data"

file_model_train = historical_data_folder / "sample_training_data.csv"
file_static_attr = processed_data_folder / "link_static_attributes.csv"
file_hist_trends = processed_data_folder / "link_historical_trends.csv"
file_sample_variance = processed_data_folder / "df_var_2023.csv"

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="type of machine learning model",
    choices=["knn", "xgboost"],
    default="knn",
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    help="Whether train the model of not",
)
args = parser.parse_args()

if __name__ == "__main__":

    # train model
    if args.train:
        model_trainer(file_model_train, arg_model=args.model)

    # dates for t-n-1 and t-n days
    for date_delta in reversed(range(0, 2)):

        previous_date, previous_date_formatted = format_dates(day_delta=date_delta + 1)
        current_date, current_date_formatted = format_dates(date_delta)
        file_raw_input = func_path_data(
            raw_data_folder, previous_date_formatted, "raw_data"
        )
        file_processed_input = func_path_data(
            processed_data_folder, previous_date_formatted, "inference_data"
        )

        # run workflow
        data_collector(raw_data_folder, previous_date_formatted, offset=date_delta * 24)
        data_processor(
            file_hist_trends,
            file_static_attr,
            file_model_train,
            file_raw_input,
            file_processed_input,
        )
        predictor(
            PATH_PREDICTIONS,
            file_processed_input,
            current_date_formatted,
            args_model=args.model,
        )
