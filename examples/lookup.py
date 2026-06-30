"""例: テーブル参照／補間 (VLOOKUP 的処理) のモンテカルロ。

ランダムに割り当てた入力に対し、テーブルを引いて値を求める典型パターンを示す。
ライブラリ側の追加機能は不要で、array スタイル + NumPy だけで書ける。

3 種類の引き方を1つの trial にまとめている:

1. 線形補間   ``np.interp``        … VLOOKUP(TRUE) 相当 (補間あり)
2. 段階参照   ``np.searchsorted``  … 区間で階段状に引く (補間なし)
3. 完全一致   ファンシーインデックス … VLOOKUP(FALSE) 相当 (整数キー)

実行方法::

    uv run python examples/lookup.py
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from polars_mc import Simulation, Uniform

# --- 参照テーブル (trial の外で一度だけ定義し、クロージャで参照する) ------------- #

# 温度 [°C] -> 材料強度 [MPa] のキャリブレーション曲線 (キーは昇順)。
TEMP_KEYS = np.array([0.0, 25.0, 50.0, 75.0, 100.0])
STRENGTH = np.array([520.0, 500.0, 470.0, 420.0, 350.0])

# 強度 -> 等級バンド。境界以上で次の等級へ上がる階段状の対応。
GRADE_EDGES = np.array([400.0, 450.0, 490.0])  # 3 本の境界 -> 4 バンド
GRADE_VALUES = np.array([1.0, 2.0, 3.0, 4.0])  # バンド番号

# 整数キーで完全一致参照するサプライヤ別の係数表。
SUPPLIER_FACTOR = np.array([1.00, 0.95, 1.10])  # supplier 0, 1, 2


def trial(
    temperature: NDArray[np.float64],
    supplier: NDArray[np.float64],
) -> dict[str, NDArray[np.generic]]:
    """温度とサプライヤから、補間・段階・完全一致の3通りでテーブルを引く。"""
    # 1) 線形補間: 温度 -> 強度 (範囲外は端値にクリップされる)
    strength = np.interp(temperature, TEMP_KEYS, STRENGTH)

    # 3) 完全一致: サプライヤ番号 (整数) -> 係数 を掛けて補正
    supplier_idx = supplier.astype(np.intp)
    strength = strength * SUPPLIER_FACTOR[supplier_idx]

    # 2) 段階参照: 補正後の強度 -> 等級バンド
    band = np.searchsorted(GRADE_EDGES, strength, side="right")
    grade = GRADE_VALUES[band]

    return {"strength": strength, "grade": grade}


def main() -> None:
    sim = Simulation(
        inputs={
            "temperature": Uniform(0.0, 100.0),
            # 0,1,2 を一様に割り当て (整数キーのソース)
            "supplier": Uniform(0.0, 3.0),
        },
        trial=trial,
        outputs={
            "strength": ["mean", "min", "max", ("q", 0.05), ("q", 0.95)],
            "grade": ["mean", "min", "max"],
        },
    )

    result = sim.run(n_trials=1_000_000, chunk_size=200_000, seed=3)
    print(result.summary())

    # 手計算での検算: 温度50°C・サプライヤ0 なら強度=470 のはず
    check = float(np.interp(50.0, TEMP_KEYS, STRENGTH)) * SUPPLIER_FACTOR[0]
    print(f"\ncheck interp(50C, supplier0) = {check:.1f} MPa")


if __name__ == "__main__":
    main()
