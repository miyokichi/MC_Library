"""例: 多角形近似した楕円同士の接触（重なり）面積 + 適応的な頂点数の細分化。

固定楕円 A の上に、中心と回転角をランダムに揺らした楕円 B を重ね、接触
（交差）面積の分布を求める。楕円は正多角形で近似するが、近似が粗いと
接触面積に系統誤差が乗る。そこで

    **交差多角形の頂点数が閾値以下だった試行だけ**、楕円の近似頂点数を
    2 倍にして再計算する（上限 ``MAX_VERTICES`` まで繰り返す）

という適応細分化を入れる。接触領域が小さく角数が少ない（＝粗い近似の
影響が大きい）試行だけ細かくやり直すので、無駄なく精度を上げられる。

適応細分化は「試行ごとに分岐する処理」だが、array スタイルのまま書ける:
条件を満たす試行のインデックスだけ抜き出し、その部分配列にベクトル化計算を
やり直せばよい（スカラの ``if`` ではなくマスクで書く）。

さらに ``main`` では 2 段階の適応を示す:

* 試行の中 …… 上記の頂点数細分化（幾何近似の精度）
* 実行の全体 … ``sim.run_until`` で平均接触面積の標準誤差が目標に達したら停止
  （サンプル数の自動調整）

Shapely が必要。実行方法::

    uv run --with shapely python examples/ellipse_intersection.py
"""

from __future__ import annotations

import numpy as np
import shapely
from numpy.typing import NDArray

from polars_mc import Normal, Simulation, Uniform

# 楕円 A（固定）: 原点中心、半径 3.0 x 1.5、回転なし
A_RX, A_RY = 3.0, 1.5
# 楕円 B: 半径 2.5 x 1.2、中心 (dx, dy) と回転角 theta は入力分布で揺らす
B_RX, B_RY = 2.5, 1.2

BASE_VERTICES = 32  # 楕円近似の初期頂点数
MAX_VERTICES = 2048  # 細分化の上限（終了保証）
REFINE_THRESHOLD = 50  # 交差多角形の頂点数がこれ以下なら細分化


def ellipse_polygons(
    cx: NDArray[np.float64],
    cy: NDArray[np.float64],
    rx: float,
    ry: float,
    angle: NDArray[np.float64],
    n_vertices: int,
) -> NDArray[np.object_]:
    """中心 (cx, cy)・半径 (rx, ry)・回転 angle の楕円を n_vertices 角形で近似。

    m 試行ぶんをまとめて作る（返り値は長さ m の geometry 配列）。
    """
    t = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    x = rx * np.cos(t)[None, :]  # (1, v)
    y = ry * np.sin(t)[None, :]
    cos_a = np.cos(angle)[:, None]  # (m, 1)
    sin_a = np.sin(angle)[:, None]
    px = cx[:, None] + cos_a * x - sin_a * y  # (m, v)
    py = cy[:, None] + sin_a * x + cos_a * y
    coords = np.stack([px, py], axis=-1)  # (m, v, 2)
    return shapely.polygons(shapely.linearrings(coords))


def trial(
    dx: NDArray[np.float64],
    dy: NDArray[np.float64],
    theta: NDArray[np.float64],
) -> dict[str, NDArray[np.generic]]:
    """1 試行 = 楕円 B を 1 配置して接触面積を求める（チャンクぶんベクトル化）。"""
    m = dx.shape[0]
    area = np.zeros(m)
    verts_used = np.full(m, BASE_VERTICES)
    zeros = np.zeros(m)

    # まだ精度が足りない試行のインデックス。最初は全試行。
    pending = np.arange(m)
    n_vertices = BASE_VERTICES
    while pending.size:
        ellipse_a = ellipse_polygons(
            zeros[pending], zeros[pending], A_RX, A_RY, zeros[pending], n_vertices
        )
        ellipse_b = ellipse_polygons(
            dx[pending], dy[pending], B_RX, B_RY, theta[pending], n_vertices
        )
        inter = shapely.intersection(ellipse_a, ellipse_b)
        area[pending] = shapely.area(inter)
        verts_used[pending] = n_vertices

        if n_vertices * 2 > MAX_VERTICES:
            break
        # 交差多角形の頂点数（閉じるための重複点を除くので -1）。
        # 凸同士の交差なので結果は単一の凸多角形（または空）。
        n_coords = shapely.get_num_coordinates(inter) - 1
        # 空交差（-1）は細分化しても空のままなので対象外。
        pending = pending[(n_coords >= 0) & (n_coords <= REFINE_THRESHOLD)]
        n_vertices *= 2

    return {"area": area, "n_vertices": verts_used}


def make_simulation() -> Simulation:
    """この例のシミュレーション定義。"""
    return Simulation(
        inputs={
            "dx": Normal(0.0, 1.5),
            "dy": Normal(0.0, 1.0),
            "theta": Uniform(0.0, np.pi),
        },
        trial=trial,
        outputs={
            # q0.00135 / q0.99865 は正規分布なら mean ± 3σ に一致する分位点
            "area": ["mean", "std", "min", "max", ("q", 0.00135), ("q", 0.99865)],
            "n_vertices": ["mean", "max"],
        },
    )


def demo_fixed_n(sim: Simulation) -> None:
    """デモ1: 固定回数の batch 実行で接触面積の分布を見る。"""
    result = sim.run(200_000, seed=42)
    print("=== デモ1: 固定 200,000 試行 ===")
    print(result.summary())

    mean = result.value("area", "mean")
    std = result.value("area", "std")
    print()
    print(f"3σ範囲（正規近似）    : [{mean - 3 * std:.4f}, {mean + 3 * std:.4f}]")
    print(
        f"3σ範囲（実分布の分位点）: "
        f"[{result.value('area', 'q0.00135'):.4f}, "
        f"{result.value('area', 'q0.99865'):.4f}]"
    )
    print(
        f"細分化の様子: 平均 {result.value('n_vertices', 'mean'):.0f} 頂点 / "
        f"最大 {result.value('n_vertices', 'max'):.0f} 頂点で計算"
    )


def demo_adaptive(sim: Simulation) -> None:
    """デモ2: 平均接触面積が目標標準誤差に達するまで試行数を自動調整（run_until）。"""
    result = sim.run_until(
        "area", target_se=1e-2, seed=42, chunk_size=25_000, max_trials=300_000
    )
    print("\n=== デモ2: 標準誤差 1e-2 に達するまで自動停止（run_until）===")
    print(result.summary())
    print(
        "→ 幾何近似（試行内の頂点細分化）とサンプル数（実行全体の収束停止）の"
        "2 段階を適応。"
    )


def main() -> None:
    sim = make_simulation()
    demo_fixed_n(sim)
    demo_adaptive(sim)


if __name__ == "__main__":
    main()
