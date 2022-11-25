# TODO: Make this file
# How to find best split?
# 1. Try a random split for each feature
# 2. Try the middle split for each feature
# 3. Try a couple (reasonable amount) of splits for each feature
#    - Random or equally split?
# How to calculate gini of two child nodes? Just average?
from __future__ import annotations

import numpy as np

from mlp.metrics import gini_score, weighted_gini
from mlp.types import IntegerArray, ScalarArray


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
        preds: IntegerArray = np.apply_along_axis(self.root.predict, 1, x)
        return preds


class Node:
    def __init__(
        self,
        depth: int,
        tree: DecisionTreeClassifier,
        impurity: float,
    ) -> None:
        self.depth = depth
        self.tree = tree
        self.impurity = impurity
        self.feature: int | None = None
        self.threshold: float | None = None
        self.left: Node | None = None
        self.right: Node | None = None
        self.label: int | None = None
        # Splitting here instead of storing self.x and self.y to allow the dataset
        # to be garbage collected

    def split(self, x: ScalarArray, y: ScalarArray) -> None:
        if not self.can_split(y):
            self.label = int(np.bincount(y).argmax())
            return
        best_split_result = self._find_best_split(x, y)
        # No splits were found, set label
        if best_split_result is None:
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

    def _find_best_split(
        self, x: ScalarArray, y: ScalarArray
    ) -> tuple[int, float, float, float, float] | None:
        features = np.arange(x.shape[1])
        splits = {}
        for split_feature in features:
            feature_vals = np.unique(x[:, split_feature])
            imps_at_feat = {}
            if len(feature_vals) < 2:
                continue
            for val1, val2 in zip(feature_vals[:-1], feature_vals[1:]):
                split_val = (val1 + val2) / 2
                mask = x[:, split_feature] < split_val
                col_mask = np.ones(x.shape[1], bool)
                col_mask[split_feature] = False
                y_left = y[mask]
                y_right = y[~mask]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                imp_left = gini_score(y_left)
                imp_right = gini_score(y_right)
                impurity = weighted_gini(
                    len(y), len(y_left), len(y_right), imp_left, imp_right
                )
                imps_at_feat[split_val] = (impurity, imp_left, imp_right)
            best_split_val = max(iter(imps_at_feat), key=lambda x: imps_at_feat[x][0])
            splits[split_feature] = (best_split_val, *imps_at_feat[best_split_val])
        if len(splits) == 0:
            return None
        best_split_feature = max(iter(splits), key=lambda x: splits[x][1])
        best_split_val, best_split_imp, imp_left, imp_right = splits[best_split_feature]
        return best_split_feature, best_split_val, best_split_imp, imp_left, imp_right

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
        if (
            len(np.unique(y)) == 1
            or (self.tree.max_depth is not None and self.depth >= self.tree.max_depth)
            or len(y) < self.tree.min_samples
        ):
            return False
        return True

    @property
    def is_leaf(self) -> bool:
        return (
            self.threshold is None and self.feature is None and self.label is not None
        )
