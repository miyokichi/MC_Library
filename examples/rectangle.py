"""例: 製造された長方形の歩留まり／品質モンテカルロ。

各試行で幅と高さを正規分布から引き、面積を計算し、最小面積の規格を満たすか
判定する。実行方法::

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
    """製品1個ぶんの試行。

    普通の関数として書く: ``width`` と ``height`` は NumPy 配列として
    (チャンクぶんをまとめて) 渡される。そのため、1試行ぶんのコードに見えても
    完全にベクトル化されて実行される。
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
            "passed": ["mean"],  # bool列の平均 ＝ 合格率
        },
    )

    result = sim.run(n_trials=10_000_000, chunk_size=500_000, seed=42)

    print(result.summary())
    print()
    print(f"yield (pass rate): {result.value('passed', 'mean'):.4%}")


if __name__ == "__main__":
    main()
