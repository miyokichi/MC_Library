"""Module-level (picklable) trials for the process-backend tests."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def rectangle_array_trial(
    w: NDArray[np.float64],
    h: NDArray[np.float64],
) -> dict[str, NDArray[np.generic]]:
    area = w * h
    return {"area": area, "passed": area >= 48.0}
