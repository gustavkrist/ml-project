from __future__ import annotations

from typing import Any

import numba as nb
import numpy as np
import numpy.typing as npt

from mlp.types import IntegerArray
from mlp.types import ScalarArray


@nb.njit(nb.float64(nb.int64[:]))
def gini_score(ys: IntegerArray) -> np.float_:
    counts = np.bincount(ys)
    class_priors = np.array(
        [count / np.sum(counts) for count in counts], dtype=np.float_
    )
    gini_score: np.float_ = 1.0 - (class_priors**2).sum()
    return gini_score


def weighted_gini_array(y_left: IntegerArray, y_right: IntegerArray) -> np.float_:
    n_l = len(y_left)
    n_r = len(y_right)
    n_t = n_l + n_r
    ret: np.float_ = weighted_gini(
        n_t, n_l, n_r, gini_score(y_left), gini_score(y_right)
    )
    return ret


@nb.njit(nb.float64(nb.int64, nb.int64, nb.int64, nb.float64, nb.float64))
def weighted_gini(
    n_t: int, n_l: int, n_r: int, imp_left: np.float_, imp_right: np.float_
) -> np.float_:
    p_l = n_l / n_t
    p_r = n_r / n_t
    return p_l * imp_left + p_r * imp_right


def accuracy_score(
    y_true: npt.NDArray[np.integer[Any]], y_pred: npt.NDArray[np.integer[Any]]
) -> float:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    acc: float = float((y_true == y_pred).sum() / y_true.shape[0])
    return acc


def ccel(s: ScalarArray, y: IntegerArray) -> float:
    """Categorical Cross-Entropy Loss"""
    # NOTE: Adding the small 1e-20 here to prevent division by zero in the log function
    ret: float = -(np.log(s[np.where(y)] + 1e-20)).sum() / y.shape[1]
    return ret


def bcel(yhat: ScalarArray, y: IntegerArray) -> float:
    """Binary Cross-Entropy Loss"""
    ret: float = -1 / len(y) * ((y @ np.log(yhat)) + ((1 - y) @ np.log(1 - yhat))).sum()
    return ret
