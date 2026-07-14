"""モンテカルロ数値積分（目標標準誤差に達したら自動停止）。

``integrate(f, a, b)`` は ``(b - a) * E[f(U)]``（U は一様分布）という
合成そのもの。記事の決定論的な数値積分（examples/whyfp_numeric.py）と
同じ「収束するまで近似を重ねる」骨格を、確率的サンプリングで実現している。

実行::

    uv run python examples/integrate.py
"""

from __future__ import annotations

import math

import numpy as np

from polars_mc import integrate


def main() -> None:
    e = integrate(lambda x: np.sin(x), 0.0, math.pi, target_se=1e-3, seed=1)
    print(f"∫_0^π sin x dx ≈ {e}   (真値 2)")

    e = integrate(lambda x: np.exp(-x * x), 0.0, 2.0, target_se=5e-4, seed=2)
    # 参考: ∫_0^2 e^(-x^2) dx = (√π/2)·erf(2) ≈ 0.882081
    true = math.sqrt(math.pi) / 2 * math.erf(2.0)
    print(f"∫_0^2 e^(-x^2) dx ≈ {e}   (真値 {true:.6f})")


if __name__ == "__main__":
    main()
