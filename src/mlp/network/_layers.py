from __future__ import annotations

from typing import Any, Callable

from mlp.network._activations import (
    leaky_relu,
    leaky_relu_der,
    relu,
    relu_der,
    sigmoid,
    sigmoid_der,
    softmax,
)


class BaseLayer:
    def __init__(self, neurons: int) -> None:
        self.neurons = neurons
        self._activation = "None"
        self.activation: Callable[[Any], Any] = self._should_not_happen
        self.activation_der: Callable[[Any], Any] = self._should_not_happen

    def _should_not_happen(self, x: Any) -> None:
        raise RuntimeError("This method should never be called")


class InputLayer(BaseLayer):
    pass


class DenseLayer(BaseLayer):
    def __init__(self, neurons: int, activation: str) -> None:
        super().__init__(neurons)
        activation = activation.lower().replace("-", "").replace("_", "")
        self._activation = activation
        self.activation: Callable[[Any], Any]
        self.activation_der: Callable[[Any], Any]
        if activation == "relu":
            self.activation = relu
            self.activation_der = relu_der
        elif activation == "sigmoid":
            self.activation = sigmoid
            self.activation_der = sigmoid_der
        elif activation == "leakyrelu":
            self.activation = leaky_relu
            self.activation_der = leaky_relu_der
        else:
            raise ValueError(
                "Invalid activation function provided. "
                'Possible options are: "relu", "leakyrelu" and "sigmoid"'
            )


class OutputLayer(BaseLayer):
    def __init__(self, neurons: int):
        super().__init__(neurons)
        self.activation: Callable[[Any], Any]
        self._multiclass = neurons > 1
        if self._multiclass:
            self.activation = softmax
            self._activation = "softmax"
        else:
            self.activation = sigmoid
            self._activation = "sigmoid"
