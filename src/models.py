from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pickle

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RepeatedKFold

import xgboost as xgb


class Model(ABC):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass

    @abstractmethod
    def cross_validation(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def load_model(self):
        pass


class KNNModel(Model):

    def __init__(self, **kwargs) -> None:
        self.model = KNeighborsRegressor(**kwargs)

    def train_model(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict_model(self, X):
        return self.model.predict(X)

    def cross_validation(self, X, y, lower_k, upper_k):
        k_values = list(range(lower_k, upper_k))
        cv_scores = [
            np.mean(
                cross_val_score(
                    KNeighborsRegressor(
                        n_neighbors=k, weights="distance", algorithm="auto", p=2
                    ),
                    X,
                    y,
                    cv=5,
                    verbose=3,
                    scoring="neg_root_mean_squared_error",
                )
            )
            for k in k_values
        ]
        optimal_k = k_values[np.argmax(cv_scores)]
        return optimal_k

    def save_model(self, path):
        saved_model = open(path, "wb")
        pickle.dump(self.model, saved_model)
        saved_model.close()

    def load_model(self, path):
        self.model = pickle.load(open(path, "rb"))


class XGBoostModel(Model):

    def __init__(self, **kwargs) -> None:
        self.model = xgb.XGBRegressor(**kwargs)

    def train_model(self, X, y):
        self.model.fit(
            X,
            y,
            verbose=2,
        )
        return self.model

    def predict_model(self, X):
        return self.model.predict(X)

    def cross_validation(self, X, y):
        cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
        self.cv_mocel = cross_val_score(
            self.model, X, y, scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1
        )
        return self.cv_model

    def save_model(self, path):
        saved_model = open(path, "wb")
        pickle.dump(self.model, saved_model)
        saved_model.close()

    def load_model(self, path):
        self.model = pickle.load(open(path, "rb"))


class EvaluationMetrics:
    def __init__(self, y, y_hat) -> None:
        self.y = y
        self.y_hat = y_hat

    def mse(self):
        return mean_squared_error(self.y, self.y_hat)

    def rmse(self):
        return np.sqrt(self.mse())
