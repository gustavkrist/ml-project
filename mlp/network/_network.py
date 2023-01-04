from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import cast

if TYPE_CHECKING:
    from typing import TypeAlias

import numpy as np
from tqdm import tqdm

from mlp.metrics import accuracy_score
from mlp.metrics import bcel
from mlp.metrics import ccel
from mlp.network._layers import DenseLayer
from mlp.network._layers import InputLayer
from mlp.network._layers import OutputLayer
from mlp.preprocessing import one_hot
from mlp.preprocessing import train_test_split
from mlp.types import Float32Array
from mlp.types import FloatArray
from mlp.types import IntArray
from mlp.types import ScalarArray
from mlp.types import UInt8Array

Layer: TypeAlias = InputLayer | DenseLayer | OutputLayer


class ForwardFeedNN:
    # pylint: disable=C0103
    def __init__(
        self,
        *layers: Layer,
        alpha: float = 10e-4,
        epochs: int = 1000,
        minibatch_size: int | None = None,
        early_stopping: int = 0,
        min_epochs: int = 0,
    ) -> None:
        self.layers = layers
        if len(layers) < 2:
            raise ValueError("At least 2 layers must be provided")
        if not (
            isinstance(layers[0], InputLayer) and isinstance(layers[-1], OutputLayer)
        ):
            raise ValueError(
                "Layer sequence must start with InputLayer and end with OutputLayer"
            )
        if not all(isinstance(layer, DenseLayer) for layer in layers[1:-1]):
            raise ValueError("All hidden layer must be of type DenseLayer")
        self.alpha = alpha
        self.epochs = epochs
        self.min_epochs = max(min_epochs, early_stopping)
        assert isinstance(self.layers[-1], OutputLayer)
        self._multiclass = self.layers[-1]._multiclass
        self.minibatch_size = minibatch_size
        self.early_stopping = early_stopping
        self.loss_history: list[np.floating[Any]] = []
        self._batch_loss_history: list[np.float_] = []
        # Glorot initialization for sigmoid, He initialization for relu
        init_const = {"leakyrelu": 2, "relu": 2, "sigmoid": 1, "softmax": 1}
        self.ws = [
            np.copy(
                np.random.normal(
                    0,
                    init_const[l_out._activation]
                    / ((l_in.neurons + l_out.neurons) / 2),
                    (l_in.neurons, l_out.neurons),
                ).astype(np.float32),
                order="F",
            )
            for l_in, l_out in zip(
                self.layers[:-1],
                self.layers[1:],
                strict=True,
            )
        ]
        self.bs = [
            np.zeros((1, layer.neurons), dtype=np.float32) for layer in self.layers[1:]
        ]

    def _get_splits(
        self, x: ScalarArray, y: ScalarArray, train_size: float
    ) -> tuple[Float32Array, Float32Array, UInt8Array, UInt8Array]:
        x_train, x_val, y_train, y_val = train_test_split(
            x, y, train_size=train_size, seed=1
        )
        x_train = np.copy(x_train.astype(np.float32))
        x_val = np.copy(x_val.astype(np.float32))
        y_train = np.copy(y_train.astype(np.uint8))
        y_val = np.copy(y_val.astype(np.uint8))
        if self._multiclass:
            y_train = one_hot(y_train)
            y_val = one_hot(y_val)
        return x_train, x_val, y_train, y_val

    def fit(self, X: ScalarArray, Y: ScalarArray, train_size: float = 0.7) -> None:
        accuracy_max = 0.0
        n_since_acc_max = 0
        x_train, x_val, y_train, y_val = self._get_splits(X, Y, train_size)
        n = x_train.shape[0]
        indices = np.arange(n)
        for i in (
            pbar := tqdm(
                range(self.epochs),
                ascii=False,
                bar_format="{desc} -{percentage:3.0f}%|{bar}| [{n_fmt}/{total_fmt}]",
            )
        ):
            np.random.shuffle(indices)
            batch_size = self.minibatch_size if self.minibatch_size is not None else n
            mini_batches: list[tuple[Float32Array, UInt8Array]] = [
                (
                    np.copy(x_train[k : k + batch_size]),
                    np.copy(y_train[k : k + batch_size]),
                )
                for k in range(0, n, batch_size)
            ]
            self._batch_loss_history = []
            for x, y in mini_batches:
                self._process_batch(x, y)
            loss = np.mean(self._batch_loss_history)
            self.loss_history.append(loss)
            accuracy = accuracy_score(y_val, self.predict(x_val))
            if accuracy <= accuracy_max:
                n_since_acc_max += 1
            else:
                accuracy_max = accuracy
                n_since_acc_max = 0
            if (
                self.early_stopping > 0
                and i >= self.min_epochs
                and n_since_acc_max == self.early_stopping
            ):
                return
            desc = (
                f"Epoch: {i+1} - Loss: {loss:.3f} - Acc: {accuracy:.2%} - "
                + f"Best acc: {accuracy_max:.2%} - Epochs since best: {n_since_acc_max}"
            )
            pbar.set_description_str(desc)

    def _process_batch(self, x: Float32Array, y: UInt8Array) -> None:
        zs, acs = self._forward(x)
        if self._multiclass:
            batch_loss = ccel(acs[-1], y)
        else:
            batch_loss = bcel(acs[-1].T, y)
        self._batch_loss_history.append(batch_loss)
        self._backpropagate(x, y, zs, acs)

    def _forward(
        self, x: Float32Array
    ) -> tuple[list[Float32Array], list[Float32Array]]:
        zs: list[Float32Array] = []
        acs: list[Float32Array] = [x]
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            zs.append(acs[i] @ w + b)
            acs.append(self.layers[i + 1].activation(zs[i]))
        return zs, acs

    def _backpropagate(
        self,
        x: Float32Array,
        y: UInt8Array,
        zs: list[Float32Array],
        acs: list[Float32Array],
    ) -> None:

        # Initialize gradients to zero
        w_grads = [np.zeros_like(w, dtype=np.float32) for w in self.ws]
        b_grads = [np.zeros_like(b, dtype=np.float32) for b in self.bs]

        # Inspired by https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/2eae26e0bdcef314dcb18f13946a94320fb28a12/network2.py#L253 # noqa: E501, B950
        # Output layer
        # This holds for binary CE loss with sigmoid or categorical CE loss with softmax
        dL_dz = acs[-1] - y
        dzo_dw = acs[-2].T @ dL_dz
        w_grads[-1] = dzo_dw
        b_grads[-1] = dL_dz.sum(axis=0, keepdims=True)

        # Subsequent layers
        for i in range(2, len(self.layers)):
            z = zs[-i]
            da_dz = self.layers[-i].activation_der(z)
            dL_dz = (dL_dz @ self.ws[-i + 1].T) * da_dz
            w_grads[-i] = acs[-i - 1].T @ dL_dz
            b_grads[-i] = dL_dz.sum(axis=0, keepdims=True)

        self.ws = [
            w - self.alpha / x.shape[0] * w_grad for w, w_grad in zip(self.ws, w_grads)
        ]
        self.bs = [
            b - self.alpha / x.shape[0] * b_grad for b, b_grad in zip(self.bs, b_grads)
        ]

    def predict_proba(self, a: ScalarArray) -> FloatArray:
        for layer, w, b in zip(self.layers[1:], self.ws, self.bs, strict=True):
            z = a @ w + b
            a = layer.activation(z)
        return cast(FloatArray, a)

    def predict(self, x: ScalarArray) -> IntArray:
        return np.argmax(self.predict_proba(x), axis=1)

    def plot_loss(self) -> None:
        import matplotlib.pyplot as plt

        plt.plot(list(range(len(self.loss_history))), self.loss_history)
        plt.show()
