import numpy as np

"""
NOTE: In the following functions:
x - predicted values from the model
y - true target values
"""


def mean_squared_error(x, y):
    return np.mean((x - y) ** 2)


def categorical_crossentropy(x, y):
    # Clip predictions to prevent log(0)
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y * np.log(x), axis=1))


VALID_LOSSES = {
    "mean_squared_error": mean_squared_error,
    # "categorical_crossentropy": categorical_crossentropy,
}
