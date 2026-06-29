"""Example: yield/quality Monte Carlo for a manufactured rectangle.

Each trial draws a width and height from normal distributions, computes the
area, and checks whether it meets a minimum-area spec. Run with::

    uv run python examples/rectangle.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polars_mc import Normal, Simulation


def trial(
    width: NDArray[np.float64],
    height: NDArray[np.float64],
) -> dict[str, NDArray[np.generic]]:
    """One manufactured part.

    Written as an ordinary function: ``width`` and ``height`` arrive as NumPy
    arrays (the whole chunk at once), so this runs fully vectorized even though
    it reads like single-trial code.
    """
    area = width * height
    passed = area >= 48.0
    return {"area": area, "passed": passed}


def main() -> None:
    sim = Simulation(
        inputs={
            "width": Normal(mean=10.0, std=0.2),
            "height": Normal(mean=5.0, std=0.1),
        },
        trial=trial,
        outputs={
            "area": ["mean", "std", "min", "max", ("q", 0.05), ("q", 0.95)],
            "passed": ["mean"],  # mean of a boolean column == pass rate
        },
    )

    result = sim.run(n_trials=10_000_000, chunk_size=500_000, seed=42)

    print(result.summary())
    print()
    print(f"yield (pass rate): {result.value('passed', 'mean'):.4%}")


if __name__ == "__main__":
    main()
