from __future__ import annotations

import warnings

import numpy as np

from mlp.types import FloatArray
from mlp.types import ScalarArray


def relu(x: ScalarArray) -> FloatArray:
    return np.where(x > 0, x, 0)


def relu_der(x: ScalarArray) -> FloatArray:
    return np.where(x < 0, 0, 1)


def leaky_relu(x: ScalarArray) -> FloatArray:
    return np.where(x < 0, 0.01 * x, x)


def leaky_relu_der(x: ScalarArray) -> FloatArray:
    return np.where(x < 0, 0.01, 1)


def sigmoid(x: ScalarArray) -> FloatArray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="overflow encountered in exp", category=RuntimeWarning
        )
        return 1 / (1 + np.exp(-x))


def sigmoid_der(x: ScalarArray) -> FloatArray:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: ScalarArray) -> FloatArray:
    # https://github.com/scipy/scipy/blob/v1.9.3/scipy/special/_logsumexp.py#L221
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
