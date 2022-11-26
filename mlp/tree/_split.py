from __future__ import annotations

import numba as nb
import numpy as np

from mlp.metrics import gini_score, weighted_gini
from mlp.types import IntegerArray, ScalarArray


@nb.njit(
    nb.types.Tuple((nb.int64, nb.float64, nb.float64, nb.float64, nb.float64))(
        nb.float64[:, :], nb.int64[:]
    ),
    parallel=True,
    nogil=True,
    boundscheck=False,
)
def find_best_split(
    x: ScalarArray, y: IntegerArray
) -> tuple[np.int_, np.float_, np.float_, np.float_, np.float_]:
    splits = np.full((x.shape[1], 4), -1.0, dtype=np.float_)
    for split_feature in nb.prange(x.shape[1]):
        feature_vals = np.unique(x[:, split_feature])
        if len(feature_vals) < 2:
            continue
        imps_at_feat = np.full((len(feature_vals), 4), -1.0, dtype=np.float_)
        for i, (val1, val2) in enumerate(zip(feature_vals[:-1], feature_vals[1:])):
            split_val = (val1 + val2) / 2
            mask = x[:, split_feature] < split_val
            col_mask = np.ones(x.shape[1], dtype=np.bool_)
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
            imps_at_feat[i, 0] = split_val
            imps_at_feat[i, 1] = impurity
            imps_at_feat[i, 2] = imp_left
            imps_at_feat[i, 3] = imp_right
        if not np.any(imps_at_feat >= 0):
            continue
        splits[split_feature] = imps_at_feat[np.argmax(imps_at_feat[:, 1])]
    if not np.any(splits >= 0):
        return np.int_(-1), np.float_(-1), np.float_(-1), np.float_(-1), np.float_(-1)
    best_split_feature = np.argmax(splits[:, 1])
    best_split_val, best_split_imp, imp_left, imp_right = splits[best_split_feature]
    return best_split_feature, best_split_val, best_split_imp, imp_left, imp_right
