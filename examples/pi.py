"""π を適応モンテカルロで推定する（目標標準誤差に達したら自動停止）。

単位円の当たり判定という「1 試行」を書くだけで、乱数サンプルの無限列 →
running 平均 → 標準誤差での収束判定、が合成コンビネータで組み上がっている。

実行::

    uv run python examples/pi.py
"""

from __future__ import annotations

import math

from mc_lib import estimate_pi


def main() -> None:
    for target_se in (1e-2, 1e-3):
        e = estimate_pi(target_se=target_se, seed=42)
        print(
            f"target_se={target_se:g}: {e}\n"
            f"    真値 π = {math.pi:.6f}, 誤差 = {abs(e.value - math.pi):.2e}"
        )


if __name__ == "__main__":
    main()
