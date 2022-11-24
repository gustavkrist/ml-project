from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mlp.types import ScalarArray


def gini_score(ys: npt.NDArray[np.integer[Any]]) -> float:
    _, counts = np.unique(ys, return_counts=True)
    class_priors = np.array([count / sum(counts) for count in counts], dtype=np.float_)
    gini_score: float = 1.0 - (class_priors**2).sum()
    return gini_score


def accuracy_score(
    y_true: npt.NDArray[np.integer[Any]], y_pred: npt.NDArray[np.integer[Any]]
) -> float:
    if len(y_true.shape) > 1:
        y_true = np.argmax(y_true, axis=1)
    acc: float = (y_true == y_pred).sum() / y_true.shape[0]
    return acc


def ccel(s: ScalarArray, y: npt.NDArray[np.integer[Any]]) -> float:
    """Categorical Cross-Entropy Loss"""
    # NOTE: Adding the small 1e-20 here to prevent division by zero in the log function
    ret: float = -(np.log(s[np.where(y)] + 1e-20)).sum() / y.shape[1]
    return ret


def bcel(yhat: ScalarArray, y: npt.NDArray[np.integer[Any]]) -> float:
    """Binary Cross-Entropy Loss"""
    ret: float = -1 / len(y) * ((y @ np.log(yhat)) + ((1 - y) @ np.log(1 - yhat))).sum()
    return ret
