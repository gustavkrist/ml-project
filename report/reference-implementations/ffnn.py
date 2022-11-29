# type: ignore
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from mlp.network import DenseLayer
from mlp.network import ForwardFeedNN
from mlp.network import InputLayer
from mlp.network import OutputLayer


class Data(Dataset):
    def __init__(self, data):
        """Load the train/test dataset"""
        data = torch.from_numpy(data)
        self.x = data[:, :784].float() / 255
        self.y = data[:, 784].long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class FFNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Input dimension is n x 748
        self.h1 = nn.Linear(784, 784 * 2)
        self.h2 = nn.Linear(784 * 2, 784)
        self.h3 = nn.Linear(784, 128)
        self.o = nn.Linear(128, 5)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, a):
        for layer in (self.h1, self.h2, self.h3):
            z = layer(a)
            a = self.leaky_relu(z)
        zo = self.o(a)
        ao = self.softmax(zo)
        return ao

    def predict(self, x):
        return torch.argmax(self.forward(x), dim=1)


def get_trained_pytorch_model(data):
    data_train, data_val = train_test_split(data, train_size=0.7)
    train = Data(data_train)
    val = Data(data_val)
    minibatch_train = DataLoader(train, batch_size=100, shuffle=True)
    batch_val = DataLoader(val, batch_size=len(val))

    clf = FFNN()

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(clf.parameters(), lr=4e-2)

    loss_history = []
    best_acc = 0
    n_since_best = 0
    epochs = 1000
    for epoch in (
        pbar := tqdm(
            range(epochs),
            ascii=False,
            bar_format="{desc} -{percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]",
        )
    ):
        epoch_loss = 0.0
        for batch in minibatch_train:
            x_batch, y_batch = batch
            prob = clf(x_batch)
            optimizer.zero_grad()
            loss = loss_func(prob, y_batch)
            epoch_loss += loss
            loss.backward()
            optimizer.step()
        x_val, y_val = next(iter(batch_val))
        preds = clf.predict(x_val)
        acc = accuracy_score(y_val, preds)
        loss_history.append(torch.mean(epoch_loss))
        if acc > best_acc:
            best_acc = acc
            n_since_best = 0
        else:
            n_since_best += 1
        if n_since_best >= 50:
            break
        desc = (
            f"Epoch: {epoch+1} - Loss: {torch.mean(epoch_loss):.3f} - Acc: {acc:.2%} - "
            + f"Best acc: {best_acc:.2%} - Epochs since best: {n_since_best}"
        )
        pbar.set_description_str(desc)
    return clf, loss_history


def get_own_trained_model(data):
    x = data[:, :784].astype(np.float_) / 255
    y = data[:, 784].astype(np.int_)

    layers = (
        InputLayer(784),
        DenseLayer(784 * 2, "leakyrelu"),
        DenseLayer(784, "leakyrelu"),
        DenseLayer(128, "leakyrelu"),
        OutputLayer(5),
    )
    ann = ForwardFeedNN(
        *layers,
        alpha=4e-2,
        epochs=1000,
        minibatch_size=100,
        early_stopping=50,
    )
    ann.fit(x, y)
    return ann


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


def main():
    data = np.load(f"{os.path.dirname(__file__)}/../../data/fashion_train.npy")
    data_train, data_val = train_test_split(data, train_size=0.7)
    py_clf, py_loss_history = get_trained_pytorch_model(np.copy(data_train))
    own_clf = get_own_trained_model(np.copy(data_train))
    x_val = data_val[:, :784].astype(np.float_) / 255
    y_val = data_val[:, 784].astype(np.int_)
    print(f"(OWN) Accuracy: {accuracy_score(y_val, own_clf.predict(x_val)):.2%}")
    batch_val = DataLoader(Data(data_val), batch_size=len(data_val))
    x_val, y_val = next(iter(batch_val))
    print(f"(PYT) Accuracy: {accuracy_score(y_val, py_clf.predict(x_val)):.2%}")
    plot_loss(py_loss_history, "PyTorch FFNN")
    plot_loss(own_clf.loss_history, "Own FFNN")


if __name__ == "__main__":
    main()
