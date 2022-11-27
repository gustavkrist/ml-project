from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

ScalarArray = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]
FloatArray = npt.NDArray[np.floating[Any]]
Float32Array = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.integer[Any]]
UInt8Array = npt.NDArray[np.uint8]
