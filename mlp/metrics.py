from __future__ import annotations

from typing import cast

import numba as nb
import numpy as np

from mlp.types import FloatArray
from mlp.types import IntArray
from mlp.types import UInt8Array


@nb.njit(nb.float32(nb.uint8[:]))
def gini_score(ys: UInt8Array) -> np.float32:
    counts = np.bincount(ys)
    class_priors = np.array(
        [count / np.sum(counts) for count in counts], dtype=np.float_
    )
    return np.float32(1 - (class_priors**2).sum())


def weighted_gini_array(y_left: UInt8Array, y_right: UInt8Array) -> np.float32:
    n_l = len(y_left)
    n_r = len(y_right)
    n_t = n_l + n_r
    return cast(
        np.float32,
        weighted_gini(n_t, n_l, n_r, gini_score(y_left), gini_score(y_right)),
    )


@nb.njit(nb.float32(nb.int64, nb.int64, nb.int64, nb.float32, nb.float32))
def weighted_gini(
    n_t: int, n_l: int, n_r: int, imp_left: np.float32, imp_right: np.float32
) -> np.float32:
    p_l = np.float32(n_l / n_t)
    p_r = np.float32(n_r / n_t)
    return p_l * imp_left + p_r * imp_right


def accuracy_score(y_true: IntArray, y_pred: IntArray) -> float:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    return float((y_true == y_pred).sum() / y_true.shape[0])


def ccel(s: FloatArray, y: IntArray) -> np.float_:
    """Categorical Cross-Entropy Loss"""
    # NOTE: Adding the small 1e-20 here to prevent division by zero in the log function
    return cast(np.float_, -(np.log(s[np.where(y)] + 1e-20)).sum() / y.shape[1])


def bcel(yhat: FloatArray, y: IntArray) -> np.float_:
    """Binary Cross-Entropy Loss"""
    return cast(
        np.float_,
        -1 / len(y) * ((y @ np.log(yhat)) + ((1 - y) @ np.log(1 - yhat))).sum(),
    )
