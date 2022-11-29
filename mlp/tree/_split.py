from __future__ import annotations

import numba as nb
import numpy as np

from mlp.metrics import gini_score
from mlp.metrics import weighted_gini
from mlp.types import Float32Array
from mlp.types import UInt8Array


@nb.njit(
    nb.types.Tuple((nb.int64, nb.float32, nb.float32, nb.float32, nb.float32))(
        nb.float32[:, :], nb.uint8[:]
    ),
    parallel=True,
    nogil=True,
    boundscheck=False,
)
def find_best_split(
    x: Float32Array, y: UInt8Array
) -> tuple[np.int_, np.float32, np.float32, np.float32, np.float32]:
    splits = np.full((x.shape[1], 4), np.inf, dtype=np.float_)
    for split_feature in nb.prange(x.shape[1]):  # pylint: disable=E1133
        feature_vals = np.unique(x[:, split_feature])
        if len(feature_vals) < 2:
            continue
        imps_at_val = np.full((len(feature_vals), 4), np.inf, dtype=np.float_)
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
            imps_at_val[i, 0] = split_val
            imps_at_val[i, 1] = impurity
            imps_at_val[i, 2] = imp_left
            imps_at_val[i, 3] = imp_right
        if np.all(np.isinf(imps_at_val)):
            continue
        splits[split_feature] = imps_at_val[np.argmin(imps_at_val[:, 1])]
    if np.all(np.isinf(splits)):
        return (
            np.int_(-1),
            np.float32(-1),
            np.float32(-1),
            np.float32(-1),
            np.float32(-1),
        )
    best_split_feature = np.argmin(splits[:, 1])
    best_split_val, best_split_imp, imp_left, imp_right = splits[best_split_feature]
    return best_split_feature, best_split_val, best_split_imp, imp_left, imp_right
