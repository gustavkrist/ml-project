from __future__ import annotations

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_fscore_support
from sklearn.tree import DecisionTreeClassifier as DTC

from mlp.metrics import accuracy_score
from mlp.tree import DecisionTreeClassifier
from mlp.types import FloatArray
from mlp.types import IntArray


def get_data() -> tuple[FloatArray, FloatArray, IntArray, IntArray]:
    data_train = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    data_test = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_test.npy")
    x_train = data_train[:, :784].astype(np.float_)
    y_train = data_train[:, 784].astype(np.int_)
    x_test = data_test[:, :784].astype(np.float_)
    y_test = data_test[:, 784].astype(np.int_)
    return x_train, x_test, y_train, y_test


def count_leaves(clf: DecisionTreeClassifier) -> int:
    leaf_count = 0
    nodes = [clf.root]
    while nodes:
        node = nodes.pop()
        if not node.is_leaf:
            assert node.left is not None and node.right is not None
            nodes.append(node.left)
            nodes.append(node.right)
        else:
            leaf_count += 1
    return leaf_count


def calc_stats(model, x_test, y_test, which):  # type: ignore
    preds = model.predict(x_test)
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        preds,
        labels=(0, 1, 2, 3, 4),
        display_labels=(0, 1, 2, 3, 4),
    )
    disp.plot(cmap="Blues")
    plt.savefig(f"figures/confusion_matrix_tree_{which}.png", dpi=400)
    results = [accuracy_score(y_test, preds)]
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, preds, average="macro"
    )
    results.extend([precision, recall, f1_score])
    leaf_count = model.get_n_leaves() if isinstance(model, DTC) else count_leaves(model)
    return pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "Leaf Count"],
            "Result": results + [leaf_count],
        }
    )


def main() -> int:
    with open("results_tree.tex", "w") as f:
        x_train, x_test, y_train, y_test = get_data()
        own_clf = DecisionTreeClassifier(
            max_depth=10, min_samples=2, early_stopping=True
        )
        start = time.monotonic()
        own_clf.fit(x_train, y_train)
        end = time.monotonic()
        print(f"(OWN) Fitting took {end-start}s")
        f.write(calc_stats(own_clf, x_test, y_test, "own").to_latex(index=False))

        x_train, x_test, y_train, y_test = get_data()
        skl_clf = DTC(max_depth=10, min_samples_split=2)
        start = time.monotonic()
        skl_clf.fit(x_train, y_train)
        end = time.monotonic()
        print(f"(SKL) Fitting took {end-start}s")
        f.write(calc_stats(skl_clf, x_test, y_test, "skl").to_latex(index=False))
    return 0


if __name__ == "__main__":
    main()
