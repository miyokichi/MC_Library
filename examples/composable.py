"""例: 部品を個別に使う（サンプリング・統計・チャンク駆動を切り分ける）。

Simulation はこれらを束ねた便利ラッパーにすぎない。ここでは同じ3段
（乱数生成 → 自前の計算 → 統計）を、各層を直接呼んで組み立てる。

実行方法::

    uv run python examples/composable.py
"""

from __future__ import annotations

import numpy as np

from polars_mc import (
    Normal,
    Summarizer,
    map_reduce_chunks,
    sample,
    sample_chunks,
    summarize,
)

INPUTS = {"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)}


def example_sampling_only() -> None:
    """(1) ランダム生成だけ: 分布から DataFrame を作る。"""
    df = sample(INPUTS, n=5, seed=42)
    print("== sample only ==")
    print(df)


def example_summarize_only() -> None:
    """(2) 統計だけ: 自前で計算した配列にチャンク合成統計をかける。"""
    # 乱数生成も面積計算も自分でやり、集計だけライブラリに任せる。
    rng = np.random.default_rng(0)
    area = rng.normal(10.0, 0.2, 1_000_000) * rng.normal(5.0, 0.1, 1_000_000)
    stats = summarize({"area": area}, {"area": ["mean", "std", ("q", 0.95)]})
    print("\n== summarize only ==")
    print(stats.summary())


def example_streaming_summary() -> None:
    """(2') 統計だけ・逐次: 自前ループで Summarizer に流し込む。"""
    s = Summarizer({"area": ["mean", "min", "max"]})
    for df in sample_chunks(INPUTS, n=1_000_000, chunk_size=200_000, seed=1):
        s.update(df.with_columns(area=df["w"] * df["h"]))
    print("\n== streaming summarize ==")
    print(s.result().summary())


def example_chunk_driver() -> None:
    """(3) チャンク駆動だけ: 独立シードのチャンクループと merge を借りる。"""

    # 各チャンクで「面積が 48 以上の件数」を数え、合算して合格率を出す。
    def count_pass(n: int, rng: np.random.Generator) -> tuple[int, int]:
        area = rng.normal(10.0, 0.2, n) * rng.normal(5.0, 0.1, n)
        return int((area >= 48.0).sum()), n

    passed, total = map_reduce_chunks(
        1_000_000,
        chunk_size=200_000,
        seed=2,
        map_fn=count_pass,
        reduce_fn=lambda a, b: (a[0] + b[0], a[1] + b[1]),
    )
    print("\n== chunk driver only ==")
    print(f"pass rate: {passed / total:.4%}")


def main() -> None:
    example_sampling_only()
    example_summarize_only()
    example_streaming_summary()
    example_chunk_driver()


if __name__ == "__main__":
    main()
