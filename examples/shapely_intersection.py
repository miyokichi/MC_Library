"""例: Shapely の幾何交差を使ったモンテカルロ。

参照ポリゴンを固定し、ランダムに揺らした中心へ可動ボックスを置いて、
重なり面積の分布を調べる。

Shapely 2.0 のベクトル化関数 (``shapely.box`` / ``shapely.intersection`` /
``shapely.area``) は geometry の NumPy 配列をまとめて処理できるため、
array スタイルにそのまま乗る (配列の1要素 ＝ 1試行)。

幾何交差は GEOS 律速で (算術よりはるかに重い)、Polars 内蔵スレッドでは
並列化できない。ここが ``backend="processes"`` の出番: チャンクは互いに
独立したCPUヘビー作業なので、スループットがコア数に比例して伸びる。
各チャンクは独立シードなので、逐次バックエンドと結果は完全に一致する。

Shapely が必要。実行方法::

    uv run --with shapely python examples/shapely_intersection.py
"""

from __future__ import annotations

import time

import numpy as np
import shapely
from numpy.typing import NDArray

from polars_mc import Normal, Simulation

# 固定の参照ポリゴン (一度だけ定義し、trial からクロージャで参照する)。
REFERENCE = shapely.box(0.0, 0.0, 10.0, 10.0)


def trial(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
) -> dict[str, NDArray[np.generic]]:
    """中心 (cx, cy) に置いた 4x4 可動ボックス1配置ぶんの試行。"""
    movable = shapely.box(cx - 2.0, cy - 2.0, cx + 2.0, cy + 2.0)
    overlap = shapely.intersection(REFERENCE, movable)
    return {"overlap_area": shapely.area(overlap)}


def main() -> None:
    sim = Simulation(
        inputs={"cx": Normal(5.0, 3.0), "cy": Normal(5.0, 3.0)},
        trial=trial,
        outputs={"overlap_area": ["mean", "min", "max", ("q", 0.5), ("q", 0.95)]},
    )

    n_trials = 1_000_000

    start = time.perf_counter()
    seq = sim.run(n_trials, chunk_size=100_000, seed=7, backend="sequential")
    seq_time = time.perf_counter() - start

    start = time.perf_counter()
    par = sim.run(n_trials, chunk_size=100_000, seed=7, backend="processes")
    par_time = time.perf_counter() - start

    print(par.summary())
    print()
    print(f"sequential: {seq_time:6.2f}s")
    print(f"processes : {par_time:6.2f}s  (speedup x{seq_time / par_time:.1f})")
    print(f"identical results: {seq.to_dict() == par.to_dict()}")


if __name__ == "__main__":
    main()
