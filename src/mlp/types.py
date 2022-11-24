from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

ScalarArray = npt.NDArray[np.floating[Any]] | npt.NDArray[np.integer[Any]]
IntegerArray = npt.NDArray[np.integer[Any]]
