# type: ignore
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from mlp.tree import DecisionTreeClassifier


def grid_search(params):
    data = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    x = data[:, :784].astype(np.float_) / 255
    y = data[:, 784].astype(np.int_)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.4)
    accuracy_scores = []
    training_times = []
    for train_idx, val_idx in sss.split(x, y):
        tree = DecisionTreeClassifier(**params)
        start = time.monotonic()
        tree.fit(x[train_idx], y[train_idx])
        end = time.monotonic()
        accuracy_scores.append(accuracy_score(y[val_idx], tree.predict(x[val_idx])))
        training_times.append(end - start)
    return accuracy_scores, training_times


def main():
    parameter_combs = {
        "max_depth": [None, None, None, 10, 10, None, 5],
        "early_stopping": [False, False, False, False, True, True, True],
        "min_samples": [2, 10, 20, 2, 2, 2, 2],
    }
    parameter_list = [{} for _ in range(len(parameter_combs["max_depth"]))]
    for param, param_val in parameter_combs.items():
        for i, v in enumerate(param_val):
            parameter_list[i][param] = v
    results = list(map(grid_search, tqdm(parameter_list)))
    with open("results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
