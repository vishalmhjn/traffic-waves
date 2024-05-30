from __future__ import annotations
from abc import ABC, abstractmethod

from sklearn.neighbors import KNeighborsRegressor


class Model(ABC):

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict_model(self):
        pass


class KNNModel(Model):

    def __init__(self, **kwargs) -> None:
        self.model = KNeighborsRegressor(**kwargs)

    def train_model(self, X, y):
        self.model.fit(X, y)
        return self.model

    def predict_model(self, X):
        return self.model.predict(X)
