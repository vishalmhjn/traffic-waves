from models import KNNModel, XGBoostModel
from dataset import DataSplitter, TimeSeriesScaler, TimeSeriesFormatter
from utils import setup_logging, predicitons_to_df

from config_model import FORECASTING_PARAMS, TRAINING_PARAMS
from config_model import (
    continous_features,
    categorical_features,
    other_columns,
    target_as_autoregressive_feature,
    target_column,
    static_features,
    dynamic_features,
)

lb, ph = (FORECASTING_PARAMS["lb"], FORECASTING_PARAMS["ph"])
model_output = TRAINING_PARAMS["model_output_dir"]

# Set up logging
logging = setup_logging("predict.log")


def predictor(predictions_folder, file_processed_input, date_formatted, args_model):
    data_object = DataSplitter(file_processed_input)
    X_formatted = data_object.df

    time_series_object = TimeSeriesScaler(
        continous_features,
        categorical_features,
        other_columns,
        target_as_autoregressive_feature,
        target_column,
    )
    _ = time_series_object.load_scaler("artifacts/minmax_scaler.gz")
    logging.info(f"Scaler successfully loaded.")

    scaled_test = time_series_object.scaler_transform(X_formatted)

    series_formatter_obj = TimeSeriesFormatter(
        lb, ph, static_features, dynamic_features, True, True
    )

    W_test, X_test, z_test = series_formatter_obj.format_data(scaled_test)
    X_test = TimeSeriesFormatter.reshape_x(X_test)
    if args_model == "knn":
        traffic_model = KNNModel()
        traffic_model.load_model(f"{model_output}/{args_model}_model")
    elif args_model == "xgboost":
        traffic_model = XGBoostModel()
        traffic_model.load_model(f"artifacts/{args_model}_model")
    logging.info(f"Model {args_model} successfully loaded.")

    y_test_hat = traffic_model.predict_model(X_test)

    df_test = predicitons_to_df(ph, z_test, y_test_hat)
    df_test.to_csv(
        predictions_folder / f"{args_model}_{date_formatted}.csv",
        index=False,
    )
    logging.info(f"Predictions for {date_formatted} successful.")
