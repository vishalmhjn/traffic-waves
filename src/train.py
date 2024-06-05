import argparse

from config_model import TRAINING_PARAMS, FORECASTING_PARAMS
from config_model import (
    continous_features,
    categorical_features,
    other_columns,
    target_as_autoregressive_feature,
    target_column,
    static_features,
    dynamic_features,
)
from config_data import prediction_date_formatted
from utils import setup_logging
from models import KNNModel, EvaluationMetrics, XGBoostModel
from dataset import DataSplitter, TimeSeriesScaler, TimeSeriesFormatter

# Set up logging
logging = setup_logging("train.log")


parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    help="type of machine learning model",
    choices=["knn", "xgboost"],
    default="knn",
)
args = parser.parse_args()

if __name__ == "__main__":

    data_object = DataSplitter(TRAINING_PARAMS["data_path"])
    X_formatted = data_object.df

    for lb, ph in [(FORECASTING_PARAMS["lb"], FORECASTING_PARAMS["ph"])]:
        det_ids = data_object.get_groups

        seed = TRAINING_PARAMS["seed"]
        validation_prop = TRAINING_PARAMS["validation_proportion"]
        test_prop = TRAINING_PARAMS["test_proportion"]

        data_object.split_groups(seed, validation_prop, test_prop)

        X_formatted_train, X_formatted_val, X_formatted_test = data_object.split_data()

        time_series_object = TimeSeriesScaler(
            continous_features,
            categorical_features,
            other_columns,
            target_as_autoregressive_feature,
            target_column,
        )

        (X_formatted_train, X_formatted_val, X_formatted_test) = [
            time_series_object.copy_target_column(df)
            for df in (X_formatted_train, X_formatted_val, X_formatted_test)
        ]

        scaler = time_series_object.scaler_fit("minmax", X_formatted_train)

        (scaled_train, scaled_val, scaled_test) = [
            time_series_object.scaler_transform(df)
            for df in (X_formatted_train, X_formatted_val, X_formatted_test)
        ]

        series_formatter_obj = TimeSeriesFormatter(
            lb, ph, static_features, dynamic_features, True, False
        )

        W_train, X_train, y_train, z_train = series_formatter_obj.format_data(
            scaled_train
        )
        W_val, X_val, y_val, z_val = series_formatter_obj.format_data(scaled_val)

        W_test, X_test, y_test, z_test = series_formatter_obj.format_data(scaled_test)

        logging.info(f"Column order: {scaled_train.columns}")

        lookback_timesteps = lb
        prediction_horizon = ph

        X_train = TimeSeriesFormatter.reshape_x(X_train)
        X_val = TimeSeriesFormatter.reshape_x(X_val)
        X_test = TimeSeriesFormatter.reshape_x(X_test)

        if args.model == "knn":
            optimal_k = 2
            traffic_model = KNNModel(
                n_neighbors=optimal_k, weights="uniform", algorithm="kd_tree", p=2
            )
        elif args.model == "xgboost":
            traffic_model = XGBoostModel(
                n_estimators=300,
                max_depth=5,
                eta=0.1,
                subsample=0.7,
                colsample_bytree=0.8,
            )
        else:
            pass

        traffic_model.train_model(X_train, y_train)

        y_train_hat = traffic_model.predict_model(X_train)
        train_rmse = EvaluationMetrics(y_train, y_train_hat).rmse()
        # logging.info("RMSE on Train Set:", train_rmse)

        y_val_hat = traffic_model.predict_model(X_val)
        val_rmse = EvaluationMetrics(y_val, y_val_hat).rmse()
        logging.info(f"RMSE on Validation Set: {val_rmse}")

        y_test_hat = traffic_model.predict_model(X_test)
        test_rmse = EvaluationMetrics(y_test, y_test_hat).rmse()
        logging.info(f"RMSE on Test Set: {test_rmse}")

        traffic_model.save_model(f"artifacts/{args.model}_model")
        time_series_object.save_scaler("artifacts/minmax_scaler.gz")
