from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import TypeAlias

import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from mlp.metrics import accuracy_score, bcel, ccel
from mlp.network._layers import DenseLayer, InputLayer, OutputLayer
from mlp.preprocessing import one_hot, train_test_split
from mlp.types import IntegerArray, ScalarArray

Layer: TypeAlias = InputLayer | DenseLayer | OutputLayer


class ForwardFeedNN:
    def __init__(
        self,
        *layers: Layer,
        alpha: float = 10e-4,
        epochs: int = 1000,
        minibatch_size: int | None = None,
        early_stopping: int = 0,
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
        assert isinstance(self.layers[-1], OutputLayer)
        self._multiclass = self.layers[-1]._multiclass
        self.minibatch_size = minibatch_size
        self.early_stopping = early_stopping
        # Glorot initialization for sigmoid, He initialization for relu
        init_const = {"leakyrelu": 2, "relu": 2, "sigmoid": 1, "softmax": 1}
        self.ws = [
            np.random.normal(
                0,
                init_const[l_out._activation] / ((l_in.neurons + l_out.neurons) / 2),
                (l_in.neurons, l_out.neurons),
            )
            for l_in, l_out in zip(
                self.layers[:-1],
                self.layers[1:],
                strict=True,
            )
        ]
        self.bs = [np.zeros((1, layer.neurons)) for layer in self.layers[1:]]

    def _get_splits(
        self, x: ScalarArray, y: IntegerArray, train_split: float
    ) -> tuple[ScalarArray, ScalarArray, IntegerArray, IntegerArray]:
        x_train, x_val, y_train, y_val = train_test_split(x, y, train_split=train_split)
        y_train = cast(IntegerArray, y_train)
        y_val = cast(IntegerArray, y_val)
        if self._multiclass:
            y_train = one_hot(y_train)
            y_val = one_hot(y_val)
        return x_train, x_val, y_train, y_val

    def fit(self, X: ScalarArray, Y: IntegerArray, train_split: float = 0.7) -> None:
        self.loss_history = []
        accuracy_max = 0.0
        n_since_acc_max = 0
        x_train, x_val, y_train, y_val = self._get_splits(X, Y, train_split)
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
            mini_batches = [
                (x_train[k : k + batch_size], y_train[k : k + batch_size])
                for k in range(0, n, batch_size)
            ]
            self._batch_loss_history: list[float] = []
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
            if self.early_stopping > 0 and n_since_acc_max == self.early_stopping:
                return
            desc = (
                f"epoch: {i+1} - Loss: {loss:.3f} - Acc: {accuracy:.2%} - "
                f"Best acc: {accuracy_max:.2%} - Epochs since best: {n_since_acc_max}"
            )
            pbar.set_description_str(desc)

    def _process_batch(self, x: ScalarArray, y: IntegerArray) -> None:
        zs, acs = self._forward(x)
        if self._multiclass:
            batch_loss = ccel(acs[-1], y)
        else:
            batch_loss = bcel(acs[-1].T, y)
        self._batch_loss_history.append(batch_loss)
        self._backpropagate(x, y, zs, acs)

    def _forward(self, x: ScalarArray) -> tuple[list[ScalarArray], list[ScalarArray]]:
        zs: list[ScalarArray] = []
        acs: list[ScalarArray] = [x]
        for i, (w, b) in enumerate(zip(self.ws, self.bs)):
            zs.append(acs[i] @ w + b)
            acs.append(self.layers[i + 1].activation(zs[i]))
        return zs, acs

    def _backpropagate(
        self,
        x: ScalarArray,
        y: IntegerArray,
        zs: list[ScalarArray],
        acs: list[ScalarArray],
    ) -> None:

        # TODO: Check that the  math is corerct

        # Initialize gradients to zero
        w_grads = [np.zeros_like(w) for w in self.ws]
        b_grads = [np.zeros_like(b) for b in self.bs]

        # Inspired by https://github.com/MichalDanielDobrzanski/DeepLearningPython/blob/2eae26e0bdcef314dcb18f13946a94320fb28a12/network2.py#L253 # noqa: E501, B950
        # Output layer
        # This holds for binary CE loss with sigmoid or categorical CE loss with softmax
        dL_dz = acs[-1] - y
        dzo_dw = acs[-2].T @ dL_dz
        w_grads[-1] = dzo_dw
        b_grads[-1] = dL_dz.sum(axis=0).reshape(1, -1)

        # Subsequent layers
        for i in range(2, len(self.layers)):
            z = zs[-i]
            da_dz = self.layers[-i].activation_der(z)
            dL_dz = (dL_dz @ self.ws[-i + 1].T) * da_dz
            w_grads[-i] = acs[-i - 1].T @ dL_dz
            b_grads[-i] = dL_dz.sum(axis=0).reshape(1, -1)

        self.ws = [
            w - self.alpha / x.shape[0] * w_grad for w, w_grad in zip(self.ws, w_grads)
        ]
        self.bs = [
            b - self.alpha / x.shape[0] * b_grad for b, b_grad in zip(self.bs, b_grads)
        ]

    def predict_proba(self, a: ScalarArray) -> ScalarArray:
        for layer, w, b in zip(self.layers[1:], self.ws, self.bs, strict=True):
            z = a @ w + b
            a = layer.activation(z)
        return a

    def predict(self, x: ScalarArray) -> npt.NDArray[np.int_]:
        preds: npt.NDArray[np.int_] = np.argmax(self.predict_proba(x), axis=1)
        return preds

    def plot_loss(self) -> None:
        import matplotlib.pyplot as plt

        plt.plot(list(range(len(self.loss_history))), self.loss_history)
        plt.show()
