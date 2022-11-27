from __future__ import annotations

import numpy as np

from mlp.metrics import gini_score
from mlp.tree._split import find_best_split
from mlp.types import IntegerArray
from mlp.types import ScalarArray


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth: int | None = None,
        early_stopping: bool = False,
        min_samples: int = 2,
    ) -> None:
        self.max_depth = max_depth
        self.early_stopping = early_stopping
        self.min_samples = min_samples

    def fit(self, x: ScalarArray, y: IntegerArray) -> None:
        impurity = gini_score(y)
        self.root = Node(0, self, impurity)
        self.root.split(x, y)

    def predict(self, x: ScalarArray) -> IntegerArray:
        return np.apply_along_axis(self.root.predict, 1, x)


class Node:
    def __init__(
        self,
        depth: int,
        tree: DecisionTreeClassifier,
        impurity: np.float_,
    ) -> None:
        self.depth = depth
        self.tree = tree
        self.impurity = impurity
        self.feature: np.int_ | None = None
        self.threshold: np.float_ | None = None
        self.left: Node | None = None
        self.right: Node | None = None
        self.label: int | None = None

    def split(self, x: ScalarArray, y: ScalarArray) -> None:
        if not self.can_split(y):
            self.label = int(np.bincount(y).argmax())
            return
        best_split_result = find_best_split(x, y)
        # No splits were found, set label
        if best_split_result[0] == -1:
            self.label = int(np.bincount(y).argmax())
            return
        split_feature, split_value, imp_split, imp_left, imp_right = best_split_result
        # Impurity did not improve, set label
        if self.tree.early_stopping and self.impurity < imp_split:
            self.label = int(np.bincount(y).argmax())
            return
        mask = x[:, split_feature] < split_value
        col_mask = np.ones(x.shape[1], bool)
        col_mask[split_feature] = False
        self.feature = split_feature
        self.threshold = split_value
        x_left = x[mask]
        y_left = y[mask]
        x_right = x[~mask]
        y_right = y[~mask]
        self.left = Node(self.depth + 1, self.tree, imp_left)
        self.right = Node(self.depth + 1, self.tree, imp_right)
        self.left.split(x_left, y_left)
        self.right.split(x_right, y_right)

    def predict(self, x: ScalarArray) -> int:
        if self.is_leaf:
            assert self.label is not None
            return self.label
        assert self.threshold is not None
        if x[self.feature] < self.threshold:
            assert self.left is not None
            return self.left.predict(x)
        assert self.right is not None
        return self.right.predict(x)

    def can_split(self, y: ScalarArray) -> bool:
        # Cannot split if pure (only one label in ys) or max depth reached
        return (
            len(np.unique(y)) == 1
            or (self.tree.max_depth is not None and self.depth >= self.tree.max_depth)
            or len(y) < self.tree.min_samples
        )

    @property
    def is_leaf(self) -> bool:
        return (
            self.threshold is None and self.feature is None and self.label is not None
        )
