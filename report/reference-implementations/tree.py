# type: ignore
from __future__ import annotations

import os
import time

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier as DTC

from mlp.metrics import accuracy_score
from mlp.tree import DecisionTreeClassifier


def get_data():
    scaler = StandardScaler()
    data = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    x = data[:, :784].astype(np.float_) / 255
    y = data[:, 784].astype(np.int_)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, train_size=0.7, random_state=1
    )
    x_train = scaler.fit_transform(x_train, y_train)
    x_val = scaler.transform(x_val)
    return x_train, x_val, y_train, y_val


def main() -> int:
    x_train, x_val, y_train, y_val = get_data()
    clf = DecisionTreeClassifier(max_depth=20, min_samples=10, early_stopping=True)
    start = time.monotonic()
    clf.fit(x_train, y_train)
    end = time.monotonic()
    print(f"Fitting took {end-start}s")
    preds = clf.predict(x_val)
    print(f"(OWN) Accuracy: {accuracy_score(y_val, preds):.2%}")

    x_train, x_val, y_train, y_val = get_data()
    clf = DTC(max_depth=20, min_samples_split=10)
    start = time.monotonic()
    clf.fit(x_train, y_train)
    end = time.monotonic()
    print(f"Fitting took {end-start}s")
    preds = clf.predict(x_val)
    print(f"(SKL) Accuracy: {accuracy_score(y_val, preds):.2%}")
    return 0


if __name__ == "__main__":
    main()
