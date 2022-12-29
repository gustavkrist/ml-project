# type: ignore
from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from mlp.network import DenseLayer
from mlp.network import ForwardFeedNN
from mlp.network import InputLayer
from mlp.network import OutputLayer


def plot_loss(loss_history, clf_name):
    _, ax = plt.subplots()
    ax.set_title(f"Epoch loss ({clf_name})")
    ax.plot(list(range(len(loss_history))), list(map(float, loss_history)))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    figpath = f"{os.path.dirname(__file__)}/figures"
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    filename = f"{clf_name.lower().replace(' ', '-')}.png"
    plt.savefig(f"{figpath}/{filename}", dpi=400, bbox_inches="tight")


def grid_search(params):
    data = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    x = data[:, :784].astype(np.float_) / 255
    y = data[:, 784].astype(np.int_)
    layers = []
    for i, layer in enumerate(param_layers := params.pop("layers")):
        if i == 0:
            layers.append(InputLayer(layer))
        elif i == len(param_layers) - 1:
            layers.append(OutputLayer(layer))
        else:
            layers.append(DenseLayer(*layer))
    sss = StratifiedShuffleSplit(n_splits=4, test_size=0.4)
    accuracy_scores = []
    auc_scores = []
    training_times = []
    loss_histories = []
    for train_idx, val_idx in sss.split(x, y):
        ann = ForwardFeedNN(*layers, **params)
        start = time.monotonic()
        ann.fit(x[train_idx], y[train_idx])
        end = time.monotonic()
        accuracy_scores.append(accuracy_score(y[val_idx], ann.predict(x[val_idx])))
        auc_scores.append(
            roc_auc_score(y[val_idx], ann.predict_proba(x[val_idx]), multi_class="ovo")
        )
        training_times.append(end - start)
        loss_histories.append(ann.loss_history)
    return accuracy_scores, auc_scores, training_times, loss_histories


def main():
    parameter_combs = {
        "layers": (
            (784, (10, "leakyrelu"), 5),
            (
                784,
                (50, "leakyrelu"),
                (50, "leakyrelu"),
                5,
            ),
            (
                784,
                (100, "leakyrelu"),
                (50, "leakyrelu"),
                (50, "leakyrelu"),
                5,
            ),
            (
                784,
                (784 * 2, "sigmoid"),
                (784, "sigmoid"),
                (128, "sigmoid"),
                5,
            ),
            (
                784,
                (784 * 2, "relu"),
                (784, "relu"),
                (128, "relu"),
                5,
            ),
            (
                784,
                (784 * 2, "leakyrelu"),
                (784, "leakyrelu"),
                (128, "leakyrelu"),
                5,
            ),
            (
                784,
                (784 * 2, "leakyrelu"),
                (784, "leakyrelu"),
                (128, "leakyrelu"),
                5,
            ),
            (
                784,
                (784 * 2, "leakyrelu"),
                (784, "leakyrelu"),
                (128, "leakyrelu"),
                5,
            ),
        ),
        "alpha": [4e-4, 8e-4, 4e-3] + [4e-2] * 5,
        "epochs": [1500, 1000, 750] + [500] * 5,
        "minibatch_size": [100] * 8,
        "early_stopping": [100] * 4 + [50] * 2 + [20] + [10],
        "min_epochs": [0] * 6 + [50] * 2,
    }
    parameter_list = [{} for _ in range(len(parameter_combs["epochs"]))]
    for param, param_val in parameter_combs.items():
        for i, v in enumerate(param_val):
            parameter_list[i][param] = v
    results = list(map(grid_search, parameter_list))
    with open("results_ffnn_.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
