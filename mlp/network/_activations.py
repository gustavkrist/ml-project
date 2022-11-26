from __future__ import annotations

import warnings

import numba as nb
import numpy as np
import numpy.typing as npt

from mlp.types import ScalarArray


@nb.njit(nb.float64[:, :](nb.float64[:, :]))
def relu(x: ScalarArray) -> ScalarArray:
    return np.maximum(x, 0)


@nb.njit(nb.int64[:, :](nb.float64[:, :]))
def relu_der(x: ScalarArray) -> npt.NDArray[np.int_]:
    return np.where(x < 0, 0, 1)


@nb.njit(nb.float64[:, :](nb.float64[:, :]))
def leaky_relu(x: ScalarArray) -> ScalarArray:
    return np.where(x < 0, 0.01 * x, x)


@nb.njit(nb.float64[:, :](nb.float64[:, :]))
def leaky_relu_der(x: ScalarArray) -> ScalarArray:
    return np.where(x < 0, 0.01, 1)


def sigmoid(x: ScalarArray) -> npt.NDArray[np.float_]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="overflow encountered in exp", category=RuntimeWarning
        )
        ret: npt.NDArray[np.float_] = 1 / (1 + np.exp(-x))
    return ret


def sigmoid_der(x: ScalarArray) -> npt.NDArray[np.float_]:
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(x: ScalarArray) -> npt.NDArray[np.float_]:
    # https://github.com/scipy/scipy/blob/v1.9.3/scipy/special/_logsumexp.py#L221
    x_max = np.amax(x, axis=1, keepdims=True)
    exp_x_shifted = np.exp(x - x_max)
    ret: npt.NDArray[np.float_] = exp_x_shifted / np.sum(
        exp_x_shifted, axis=1, keepdims=True
    )
    return ret
