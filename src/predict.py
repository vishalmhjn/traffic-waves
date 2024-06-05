from pathlib import Path
import argparse

from config_model import FORECASTING_PARAMS
from config_model import (
    continous_features,
    categorical_features,
    other_columns,
    target_as_autoregressive_feature,
    target_column,
    static_features,
    dynamic_features,
)
from config_data import prediction_date_formatted, file_processed_input

from models import KNNModel, XGBoostModel
from dataset import DataSplitter, TimeSeriesScaler, TimeSeriesFormatter
from utils import setup_logging, predicitons_to_df

lb, ph = (FORECASTING_PARAMS["lb"], FORECASTING_PARAMS["ph"])

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="type of machine learning model",
    choices=["knn", "xgboost"],
    default="knn",
)
args = parser.parse_args()

# Set up logging
logging = setup_logging("predict.log")

if __name__ == "__main__":

    data_object = DataSplitter(file_processed_input)
    X_formatted = data_object.df

    time_series_object = TimeSeriesScaler(
        continous_features,
        categorical_features,
        other_columns,
        target_as_autoregressive_feature,
        target_column,
    )
    scaler = time_series_object.load_scaler("artifacts/minmax_scaler.gz")
    scaled_test = time_series_object.scaler_transform(X_formatted)

    series_formatter_obj = TimeSeriesFormatter(
        lb, ph, static_features, dynamic_features, True, True
    )

    W_test, X_test, z_test = series_formatter_obj.format_data(scaled_test)
    X_test = TimeSeriesFormatter.reshape_x(X_test)

    if args.model == "knn":
        traffic_model = KNNModel()
        traffic_model.load_model(f"artifacts/{args.model}_model")
    elif args.model == "xgboost":
        traffic_model = XGBoostModel()
        traffic_model.load_model(f"artifacts/{args.model}_model")

    y_test_hat = traffic_model.predict_model(X_test)

    df_test = predicitons_to_df(ph, z_test, y_test_hat)
    df_test.to_csv(
        Path("..") / "predictions" / f"knn_{prediction_date_formatted}.csv", index=False
    )
