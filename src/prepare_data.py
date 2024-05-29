from itertools import groupby
from operator import itemgetter
from typing import List, Tuple
import numpy as np
import pandas as pd


# def preprocess_data(
#     X_list: List[np.ndarray],
#     W_list: List[np.ndarray],
#     lookback_timesteps: int,
#     dyn_to_static: List[str],
#     y_list: List[np.ndarray] = None,
# ) -> Tuple[np.array, np.array, np.array]:
#     """
#     Preprocesses data for input to a model.
#     Can be used for both training (with y) and inference (without y).

#     Args:
#         X_list (List[np.ndarray]): List of numpy arrays containing dynamic features.
#         W_list (List[np.ndarray]): List of numpy arrays containing static features.
#         lookback_timesteps (int): Number of timesteps to look back in the sequence.
#         dyn_to_static (List[str]): List of strings representing dynamic features to be treated as static.
#         y_list (List[np.ndarray], optional): List of numpy arrays containing target values. Defaults to None.

#     Returns:
#         Tuple[np.array, np.array, np.array]: Preprocessed data as numpy arrays (X, W, and optionally y).
#     """
#     X = np.array(X_list, dtype="float64")
#     W = np.array(W_list, dtype="float64")

#     W = np.concatenate(
#         [
#             W,
#             X[:, int(np.floor(lookback_timesteps / 2)), : len(dyn_to_static)].reshape(
#                 X.shape[0], -1
#             ),
#         ],
#         axis=1,
#     )
#     X = X[:, :, len(dyn_to_static) :]

#     W[:, -1] = np.round(W[:, -1])
#     W[:, -2] = np.round(W[:, -2])

#     if y_list is not None:
#         y = np.array(y_list, dtype="float64")
#         return X, y, W
#     else:
#         return X, W


def split_sequences(
    sequences: np.ndarray,
    static: np.ndarray,
    id_det: int,
    n_steps: int,
    p_horizon: int = None,
    auto_regressive: bool = False,
    inference: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a multivariate sequence into samples for training or inference.

    Args:
        sequences (np.ndarray): Multivariate sequence to split.
        static (np.ndarray): Static values for the sequence.
        id_det (int): Identifier for the sequence.
        n_steps (int): Number of time steps for each sample.
        p_horizon (int, optional): Prediction horizon for each sample. Not required for inference.
        auto_regressive (bool, optional): Whether the sequence is auto-regressive. Defaults to False.
        inference (bool, optional): Whether the function is used for inference. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Tuple containing arrays (W, X, y, z) where:
            - W is the static values,
            - X is the input sequences,
            - y is the output sequences (only for training),
            - z is the identifier.
    """
    W, X, y, z = list(), list(), list(), list()

    if inference:
        indices = [0]
    else:
        indices = range(len(sequences))

    for i in indices:
        end_ix = i + n_steps

        if not inference and (end_ix + p_horizon > len(sequences)):
            break

        if not auto_regressive:
            seq_x = sequences[i:end_ix, :-1]
        else:
            seq_x = sequences[i:end_ix, :]

        X.append(seq_x)
        W.append(static)
        z.append(id_det)

        if not inference:
            seq_y = sequences[end_ix : end_ix + p_horizon, -1]
            y.append(seq_y)

    if inference:
        return np.array(W), np.array(X), np.array(z)
    else:
        return np.array(W), np.array(X), np.array(y), np.array(z)


def format_data(
    df: pd.DataFrame,
    lookback_timesteps: int,
    features_static: List[str],
    features_dynamic: List[str],
    prediction_horizon: int = None,
    auto_regressive: bool = False,
    inference: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Format the data for training or inference.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        lookback_timesteps (int): Number of lookback timesteps.
        features_static (List[str]): List of static features.
        features_dynamic (List[str]): List of dynamic features.
        prediction_horizon (int, optional): Prediction horizon. Required for training.
        auto_regressive (bool, optional): Whether the sequence is auto-regressive. Defaults to False.
        inference (bool, optional): Whether the function is used for inference. Defaults to False.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[int]]:
        Tuple containing lists of arrays (W_list, X_list, y_list, z_list) where:
            - W_list contains static values,
            - X_list contains input sequences,
            - y_list contains output sequences (only for training),
            - z_list contains identifiers.
    """
    W_list, X_list, y_list, z_list = list(), list(), list(), list()
    for i in df.paris_id.unique():
        temp = df[df.paris_id == i]
        temp = temp.sort_values(by="time_idx")
        temp.index = temp.time_idx

        w = np.array(temp[features_static].drop_duplicates())[0]
        for k, g in groupby(enumerate(list(temp.index)), lambda ix: ix[0] - ix[1]):
            temp_list = list(map(itemgetter(1), g))

            if len(temp_list) >= lookback_timesteps:
                temp_df = temp.loc[temp_list, features_dynamic]
                if inference:
                    W, X, z = split_sequences(
                        np.array(temp_df),
                        w,
                        i,
                        lookback_timesteps,
                        auto_regressive=auto_regressive,
                        inference=True,
                    )
                    W_list.extend(W)
                    X_list.extend(X)
                    z_list.extend(z)
                else:
                    W, X, y, z = split_sequences(
                        np.array(temp_df),
                        w,
                        i,
                        lookback_timesteps,
                        prediction_horizon,
                        auto_regressive=auto_regressive,
                        inference=False,
                    )
                    W_list.extend(W)
                    X_list.extend(X)
                    y_list.extend(y)
                    z_list.extend(z)
    if inference:
        return W_list, X_list, z_list
    else:
        return W_list, X_list, y_list, z_list


# def split_data(
#     X: np.ndarray, W: np.ndarray, y: np.ndarray = None
# ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Splits the input data into categorical and continuous parts.
#     Can be used for both training (with y) and inference (without y).

#     Args:
#         X (np.ndarray): Array containing input features.
#         W (np.ndarray): Array containing static features.
#         y (np.ndarray, optional): Array containing target values. Defaults to None.

#     Returns:
#         Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
#             - cat_stat (np.ndarray): Array of categorical static features.
#             - cont_stat (np.ndarray): Array of continuous static features.
#             - cat_dyn (np.ndarray): Array of categorical dynamic features.
#             - cont_dyn (np.ndarray): Array of continuous dynamic features.
#     """
#     static_cols = 3
#     cat_stat = W[:, static_cols:]
#     cont_stat = W[:, :static_cols]
#     cat_dyn = X[:, :, :0]
#     cont_dyn = X[:, :, 0:]

#     return cat_stat, cont_stat, cat_dyn, cont_dyn


def scale_and_create_df(
    X_formatted,
    scaler,
    continous_feature_columns,
    categorical_feature_columns,
    other_columns,
    target_as_autoregressive_feature_name,
    target_column_name,
):
    """
    Scales the continuous feature columns, creates a DataFrame with scaled features,
    and assigns categorical features and other columns to the scaled DataFrame.

    Args:
            X_formatted (pd.DataFrame): Original DataFrame with features.
            scaler: Scaler object to transform the continuous feature columns.
            continous_feature_columns (List[str]): List of column names for continuous features.
            categorical_feature_columns (List[str]): List of column names for categorical features.
            other_columns (List[str]): List of column names for other features.

    Returns:
            pd.DataFrame: DataFrame containing scaled features.
    """
    scaled_features = scaler.transform(X_formatted[continous_feature_columns].values)

    # Create DataFrame with scaled features
    scaled_features_df = pd.DataFrame(
        scaled_features, index=X_formatted.index, columns=continous_feature_columns
    )

    # Assign categorical features and other columns to scaled DataFrame
    scaled_features_df[categorical_feature_columns] = X_formatted[
        categorical_feature_columns
    ]
    scaled_features_df[other_columns] = X_formatted[other_columns]

    # Assign 'q' column to scaled DataFrame
    scaled_features_df[target_column_name] = X_formatted[
        target_as_autoregressive_feature_name
    ]

    return scaled_features_df
