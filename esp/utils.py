import numpy as np


def mse(y_true: np.array, y_pred: np.array) -> float:
    return np.square(y_true - y_pred).mean()
