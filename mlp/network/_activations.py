from __future__ import annotations

import warnings
from typing import cast

import numexpr as ne
import numpy as np

from mlp.types import Float32Array
from mlp.types import FloatArray
from mlp.types import ScalarArray


def relu(x: ScalarArray) -> FloatArray:
    return cast(FloatArray, ne.evaluate("where(x > 0, x, 0)", casting="no"))


def relu_der(x: ScalarArray) -> FloatArray:
    a = np.float32(0.0)  # noqa: F841
    b = np.float32(0.0)  # noqa: F841
    return cast(Float32Array, ne.evaluate("where(x < 0, a, b)", casting="no"))


def leaky_relu(x: ScalarArray) -> FloatArray:
    a = np.float32(0.01)  # noqa: F841
    return cast(FloatArray, ne.evaluate("where(x < 0, a * x, x)", casting="no"))


def leaky_relu_der(x: ScalarArray) -> FloatArray:
    a = np.float32(0.01)  # noqa: F841
    return cast(FloatArray, ne.evaluate("where(x < 0, a, 1)", casting="no"))


def sigmoid(x: ScalarArray) -> FloatArray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="overflow encountered in exp", category=RuntimeWarning
        )
        return cast(FloatArray, 1 / (1 + np.exp(-x)))


def sigmoid_der(x: ScalarArray) -> FloatArray:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: ScalarArray) -> FloatArray:
    # https://github.com/scipy/scipy/blob/v1.9.3/scipy/special/_logsumexp.py#L221
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    return cast(
        FloatArray, exp_x_shifted / np.sum(exp_x_shifted, axis=1, keepdims=True)
    )
