from pathlib import Path
from config_model import *
from models import KNNModel
from dataset import DataSplitter, TimeSeriesScaler, TimeSeriesFormatter
from utils import setup_logging, predicitons_to_df

inference_date = INFERENCE_INPUT_DATE
inference_file = file_processed_input

lb, ph = (MODEL_PARAMS["lb"], MODEL_PARAMS["ph"])
# Set up logging
logging = setup_logging("predict.log")

if __name__ == "__main__":

    data_object = DataSplitter(inference_file)
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

    traffic_model = KNNModel()
    traffic_model.load_model("artifacts/knn_model")

    y_test_hat = traffic_model.predict_model(X_test)

    df_test = predicitons_to_df(ph, z_test, y_test_hat)
    df_test.to_csv(
        Path("..") / "predictions" / f"knn_{INFERENCE_PREDICTION_DATE}.csv", index=False
    )
