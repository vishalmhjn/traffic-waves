# Standard library imports
import random
from itertools import repeat

# Third-party imports
import numpy as np
import pandas as pd

# from tqdm import tqdm
from utils import setup_logging
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

from config_model import *
from prepare_data import (
    format_data,
    scale_and_create_df,
)

inference_file = file_processed_input


from losses import *


# Set up logging
logging = setup_logging("logfile_model.log")


if __name__ == "__main__":
    X_formatted = pd.read_csv(data_path)
    # X_formatted = X_formatted[X_formatted["month"] >= 4]

    for lb, ph in [(MODEL_PARAMS["lb"], MODEL_PARAMS["ph"])]:
        det_ids = list(X_formatted.paris_id.unique())

        n_test = int(CONFIG["test_proportion"] * len(det_ids))

        seed = CONFIG["seed"]
        test_ids = list(random.sample(det_ids, n_test))
        for seed in range(0, 1):
            random.seed(seed)

            non_test_ids = []

            for i in det_ids:
                if i not in test_ids:
                    non_test_ids.append(i)

            random.seed(seed)
            val_ids = list(
                random.sample(
                    non_test_ids, int(CONFIG["validation_proportion"] * len(det_ids))
                )
            )
            new_train_ids = []
            for i in non_test_ids:
                if i not in val_ids:
                    new_train_ids.append(i)
            train_ids = new_train_ids

            X_formatted_train = X_formatted[X_formatted.paris_id.isin(train_ids)]
            X_formatted_test = X_formatted[X_formatted.paris_id.isin(test_ids)]
            X_formatted_val = X_formatted[X_formatted.paris_id.isin(val_ids)]

            X_formatted_train.loc[X_formatted_train.index, target_column] = (
                X_formatted_train[target_as_autoregressive_feature]
            )
            X_formatted_val.loc[X_formatted_val.index, target_column] = X_formatted_val[
                target_as_autoregressive_feature
            ]
            X_formatted_test.loc[X_formatted_test.index, target_column] = (
                X_formatted_test[target_as_autoregressive_feature]
            )

            # Initialize the StandardScaler
            scaler = StandardScaler()

            # Fit scaler to training data
            scaler.fit(X_formatted_train[continous_features].values)

            # Scale and create DataFrames for each dataset
            scaled_features_df_train = scale_and_create_df(
                X_formatted_train,
                scaler,
                continous_features,
                categorical_features,
                other_columns,
                target_as_autoregressive_feature,
                target_column,
            )
            scaled_features_df_val = scale_and_create_df(
                X_formatted_val,
                scaler,
                continous_features,
                categorical_features,
                other_columns,
                target_as_autoregressive_feature,
                target_column,
            )
            scaled_features_df_test = scale_and_create_df(
                X_formatted_test,
                scaler,
                continous_features,
                categorical_features,
                other_columns,
                target_as_autoregressive_feature,
                target_column,
            )

            logging.info(f"Column order: {scaled_features_df_train.columns}")

            lookback_timesteps = lb
            prediction_horizon = ph

            W_list_train, X_list_train, y_list_train, z_list_train = format_data(
                scaled_features_df_train,
                lookback_timesteps,
                static_features,
                dynamic_features,
                prediction_horizon,
            )

            W_list_val, X_list_val, y_list_val, z_list_val = format_data(
                scaled_features_df_val,
                lookback_timesteps,
                static_features,
                dynamic_features,
                prediction_horizon,
            )
            W_list_test, X_list_test, y_list_test, z_list_test = format_data(
                scaled_features_df_test,
                lookback_timesteps,
                static_features,
                dynamic_features,
                prediction_horizon,
            )
            W_array_train, X_array_train, y_array_train, z_array_train = (
                np.array(W_list_train),
                np.array(X_list_train),
                np.array(y_list_train),
                np.array(z_list_train),
            )

            W_array_val, X_array_val, y_array_val, z_array_val = (
                np.array(W_list_val),
                np.array(X_list_val),
                np.array(y_list_val),
                np.array(z_list_val),
            )

            W_array_test, X_array_test, y_array_test, z_array_test = (
                np.array(W_list_test),
                np.array(X_list_test),
                np.array(y_list_test),
                np.array(z_list_test),
            )

            def reshape_x(W, X):
                X_reshaped = np.reshape(X, (X.shape[0], -1))
                # combined_array = np.hstack((W, X_reshaped))
                return X_reshaped  # combined_array

            train_X = reshape_x(W_array_train, X_array_train)
            val_X = reshape_x(W_array_val, X_array_val)
            test_X = reshape_x(W_array_test, X_array_test)

            k_values = list(range(1, 12))

            from sklearn.model_selection import cross_val_score

            cv_scores = [
                np.mean(
                    cross_val_score(
                        KNeighborsRegressor(
                            n_neighbors=k, weights="distance", algorithm="auto", p=2
                        ),
                        train_X,
                        y_array_train,
                        cv=5,
                        verbose=3,
                        scoring="neg_root_mean_squared_error",
                    )
                )
                for k in k_values
            ]

            optimal_k = k_values[np.argmax(cv_scores)]
            print(f"Optimal number of neighbors: {optimal_k}")

            neigh = KNeighborsRegressor(
                n_neighbors=optimal_k, weights="uniform", algorithm="kd_tree", p=2
            )
            neigh.fit(train_X, y_array_train)
            # Predict on the train set
            print(y_array_train)

            train_predictions = neigh.predict(train_X)
            print(train_predictions)
            # Evaluate the model using mean squared error
            train_mse = mean_squared_error(y_array_train, train_predictions)
            print("RMSE on Train Set:", np.sqrt(train_mse))

            # Predict on the validation set
            val_predictions = neigh.predict(val_X)
            # Evaluate the model using mean squared error
            val_mse = mean_squared_error(y_array_val, val_predictions)
            print("RMSE on Validation Set:", np.sqrt(val_mse))

            # Predict on the validation set
            test_predictions = neigh.predict(test_X)

            # Evaluate the model using mean squared error
            test_mse = mean_squared_error(y_array_test, test_predictions)
            print("RMSE on Test Set:", np.sqrt(test_mse))

            inference_date = INFERENCE_INPUT_DATE

            X_formatted_test = pd.read_csv(inference_file)

            scaled_features_df_test = scale_and_create_df(
                X_formatted_test,
                scaler,
                continous_features,
                categorical_features,
                other_columns,
                target_as_autoregressive_feature,
                target_column,
            )

            W_list_test, X_list_test, z_list_test = format_data(
                scaled_features_df_test,
                lookback_timesteps,
                static_features,
                dynamic_features,
                inference=True,
            )

            W_array_test, X_array_test, y_array_test, z_array_test = (
                np.array(W_list_test),
                np.array(X_list_test),
                np.array(y_list_test),
                np.array(z_list_test),
            )
            test_X = reshape_x(W_array_test, X_array_test)
            test_predictions = neigh.predict(test_X)

            df_test_diff = pd.DataFrame(
                {
                    "paris_id": [
                        x
                        for x in z_array_test
                        for _ in repeat(None, prediction_horizon)
                    ]
                }
            )
            df_test_diff["time_idx"] = np.tile(
                np.arange(prediction_horizon), len(z_array_test)
            )
            df_test_diff["preds"] = np.ravel(test_predictions)
            df_test_diff["preds"] = df_test_diff["preds"].astype(int)
            df_test_diff.to_csv(
                f"../predictions/knn_{INFERENCE_PREDICTION_DATE}.csv", index=False
            )
