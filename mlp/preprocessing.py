from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import chain
from itertools import groupby
from typing import cast

import numpy as np

from mlp.types import IntArray
from mlp.types import ScalarArray
from mlp.types import UInt8Array


def one_hot(y: IntArray, categories: Sequence[int] | None = None) -> UInt8Array:
    label: np.int_ | int = 0
    if categories is not None:
        label = cast(int, label)
        y_one_hot = np.zeros((y.shape[0], len(categories)), dtype=np.uint8, order="F")
        for i, label in enumerate(categories):
            y_one_hot[np.where(y == label), i] = 1
    else:
        label = cast(np.int_, label)
        unique = np.unique(y)
        y_one_hot = np.zeros((y.shape[0], unique.shape[0]), dtype=np.uint8, order="F")
        for i, label in enumerate(unique):
            y_one_hot[np.where(y == label), i] = 1
    return y_one_hot


def train_test_split(
    *arrays: ScalarArray, train_split: float, seed: int | None = None
) -> Sequence[ScalarArray]:
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided as a positional argument")
    shapes = groupby(map(lambda x: x.shape[0], arrays))
    if next(shapes, True) and next(shapes, False):
        raise ValueError(
            "The provided arrays must have the same shape along the first dimension"
        )
    n = arrays[0].shape[0]
    indices = np.arange(n)
    train_indices = np.random.default_rng(seed=seed).choice(
        indices, replace=False, size=math.floor(train_split * n)
    )
    test_mask = np.ones(n, dtype=bool)
    test_mask[train_indices] = False
    return tuple(chain(*((arr[train_indices], arr[test_mask]) for arr in arrays)))
