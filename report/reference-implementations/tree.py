# type: ignore
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.tree import DecisionTreeClassifier as DTC

from mlp.metrics import accuracy_score
from mlp.tree import DecisionTreeClassifier


def get_data():
    data_train = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    data_test = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    x_train = data_train[:, :784].astype(np.float_)
    y_train = data_train[:, 784].astype(np.int_)
    x_test = data_test[:, :784].astype(np.float_)
    y_test = data_test[:, 784].astype(np.int_)
    return x_train, x_test, y_train, y_test


def main() -> int:
    x_train, x_test, y_train, y_test = get_data()
    clf = DecisionTreeClassifier(max_depth=10, min_samples=2, early_stopping=True)
    start = time.monotonic()
    clf.fit(x_train, y_train)
    end = time.monotonic()
    print(f"Fitting took {end-start}s")
    preds = clf.predict(x_test)
    print(f"(OWN) Accuracy: {accuracy_score(y_test, preds):.2%}")

    x_train, x_test, y_train, y_test = get_data()
    clf = DTC(max_depth=10, min_samples_split=2)
    start = time.monotonic()
    clf.fit(x_train, y_train)
    end = time.monotonic()
    print(f"Fitting took {end-start}s")
    preds = clf.predict(x_test)
    print(f"(SKL) Accuracy: {accuracy_score(y_test, preds):.2%}")
    return 0


if __name__ == "__main__":
    main()
