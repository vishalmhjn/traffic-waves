from abc import ABC, abstractmethod
import random
from itertools import groupby
from operator import itemgetter
from typing import List, Tuple
import numpy as np
import pandas as pd

import joblib
from sklearn.preprocessing import StandardScaler


class Dataset:
    def __init__(self, path) -> None:
        self.path = path
        self.df = self.read_data()
        super().__init__()

    def read_data(self):
        df = pd.read_csv(self.path)
        return df

    @property
    def get_groups(self):
        return list(self.df.paris_id.unique())


class DataSplitter(Dataset):
    def __init__(self, path) -> None:
        super().__init__(path)

    def split_groups(self, seed, val_prop, test_prop):

        len_groups = len(self.get_groups)

        n_test = int(test_prop * len_groups)
        self.test_ids = list(random.sample(self.get_groups, n_test))
        non_test_ids = []

        random.seed(seed)
        for i in self.get_groups:
            if i not in self.test_ids:
                non_test_ids.append(i)

        self.val_ids = list(random.sample(non_test_ids, int(val_prop * len_groups)))
        new_train_ids = []
        for i in non_test_ids:
            if i not in self.val_ids:
                new_train_ids.append(i)
        self.train_ids = new_train_ids

    def split_data(self):
        train = self.df[self.df.paris_id.isin(self.train_ids)]
        test = self.df[self.df.paris_id.isin(self.test_ids)]
        val = self.df[self.df.paris_id.isin(self.val_ids)]
        return train, test, val


class DataScaler(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def scaler_fit(self):
        pass

    @abstractmethod
    def scaler_transform(self):
        pass

    @abstractmethod
    def save_scaler(self):
        pass

    @abstractmethod
    def load_scaler(self):
        pass


class TimeSeriesScaler(DataScaler):

    def __init__(
        self,
        continous_features,
        categorical_features,
        other_columns,
        original_target_column,
        duplicated_target_column,
    ) -> None:
        self.continous_features = continous_features
        self.categorical_features = categorical_features
        self.other_columns = other_columns
        self.original_target_column = original_target_column
        self.duplicated_target_column = duplicated_target_column
        super().__init__()

    def copy_target_column(self, _df):
        _df.loc[_df.index, self.duplicated_target_column] = _df[
            self.original_target_column
        ]
        return _df

    def scaler_fit(self, scaler_type, X):
        if scaler_type == "minmax":
            self.scaler = StandardScaler()
        else:
            raise NotImplementedError
        self.scaler.fit(X[self.continous_features].values)
        return self.scaler

    def scaler_transform(self, X):
        scaled_features = self.scaler.transform(X[self.continous_features].values)
        scaled_features_df = pd.DataFrame(
            scaled_features, index=X.index, columns=self.continous_features
        )
        scaled_features_df[self.categorical_features] = X[self.categorical_features]
        scaled_features_df[self.other_columns] = X[self.other_columns]
        # Assign 'q' column to scaled DataFrame
        scaled_features_df[self.duplicated_target_column] = X[
            self.original_target_column
        ]
        return scaled_features_df

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        scaler = joblib.load(path)
        self.scaler = scaler


class TimeSeriesFormatter:

    def __init__(
        self,
        lookback_timesteps,
        prediction_horizon,
        features_static,
        features_dynamic,
        auto_regressive,
        inference,
    ) -> None:
        self.lb = lookback_timesteps
        self.ph = prediction_horizon
        self.static_fs = features_static
        self.dynamic_fs = features_dynamic
        self.auto_regressive = auto_regressive
        self.inference = inference

    @staticmethod
    def reshape_x(X, W=None, use_static=False):
        X_reshaped = np.reshape(X, (X.shape[0], -1))
        if use_static:
            X_reshaped = np.hstack((W, X_reshaped))
        return X_reshaped

    def split_sequences(
        self,
        sequences: np.ndarray,
        static: np.ndarray,
        id_det: int,
    ):
        W, X, y, z = list(), list(), list(), list()

        if self.inference:
            indices = [0]
        else:
            indices = range(len(sequences))

        for i in indices:
            end_ix = i + self.lb

            if not self.inference and (end_ix + self.ph > len(sequences)):
                break

            if not self.auto_regressive:
                seq_x = sequences[i:end_ix, :-1]
            else:
                seq_x = sequences[i:end_ix, :]

            X.append(seq_x)
            W.append(static)
            z.append(id_det)

            if not self.inference:
                seq_y = sequences[end_ix : end_ix + self.ph, -1]
                y.append(seq_y)

        if self.inference:
            return np.array(W), np.array(X), np.array(z)
        else:
            return np.array(W), np.array(X), np.array(y), np.array(z)

    def format_data(self, df):

        W_list, X_list, y_list, z_list = list(), list(), list(), list()
        for i in df.paris_id.unique():
            temp = df[df.paris_id == i]
            temp = temp.sort_values(by="time_idx")
            temp.index = temp.time_idx

            w = np.array(temp[self.static_fs].drop_duplicates())[0]
            for k, g in groupby(enumerate(list(temp.index)), lambda ix: ix[0] - ix[1]):
                temp_list = list(map(itemgetter(1), g))

                if len(temp_list) >= self.lb:
                    temp_df = temp.loc[temp_list, self.dynamic_fs]
                    if self.inference:
                        W, X, z = self.split_sequences(np.array(temp_df), w, i)
                        W_list.extend(W)
                        X_list.extend(X)
                        z_list.extend(z)
                    else:
                        W, X, y, z = self.split_sequences(np.array(temp_df), w, i)
                        W_list.extend(W)
                        X_list.extend(X)
                        y_list.extend(y)
                        z_list.extend(z)
        if self.inference:
            return np.array(W_list), np.array(X_list), np.array(z_list)
        else:
            return (
                np.array(W_list),
                np.array(X_list),
                np.array(y_list),
                np.array(z_list),
            )
