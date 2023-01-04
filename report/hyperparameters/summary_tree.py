from __future__ import annotations

import json

import numpy as np

with open("results_tree.json") as f:
    results = json.load(f)


for i, result in enumerate(results):
    acc_scores, train_times = result
    print(f"{i+1}: {np.mean(acc_scores):.2%} Acc, {np.mean(train_times):.2f}s to train")
