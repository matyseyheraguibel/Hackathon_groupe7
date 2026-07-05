from abc import ABC, abstractmethod

class PriceModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("Bro déclaque toi sur fit")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("Bro déclaque toi sur predict")