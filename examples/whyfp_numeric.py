"""記事「なぜ関数プログラミングは重要か」の数値例を Stream コンビネータで再現する。

John Hughes の記事に出てくる 3 つの数値アルゴリズム

* ニュートン・ラフソン法の平方根
* 数値微分（誤差項の消去による改良つき）
* 数値積分（区間再帰による改良列）

を、いずれも ``mc_lib.stream.Stream`` の高階コンビネータと遅延無限列だけで書く。
どれも「無限の近似列を作り → 部品を合成して改良し → 収束したら打ち切る」という
同じ骨格でできていることが、この DSL の本質（＝記事の本質）である。

実行::

    uv run python examples/whyfp_numeric.py
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator

from mc_lib.stream import Stream

# --------------------------------------------------------------------------
# 1. ニュートン・ラフソン法の平方根
#    sqrt N = within eps (repeat (next N) x0)
# --------------------------------------------------------------------------


def newton_sqrt(n: float, *, eps: float = 1e-12, x0: float = 1.0) -> float:
    """``next x = (x + N/x) / 2`` の反復列を収束するまでたどる。"""
    return Stream.iterate(lambda x: (x + n / x) / 2, x0).within(eps)


# --------------------------------------------------------------------------
# 2. 数値微分
#    easydiff → differentiate（h を半分にする無限列）→ elimerror/improve で改良
# --------------------------------------------------------------------------


def easydiff(f: Callable[[float], float], x: float, h: float) -> float:
    return (f(x + h) - f(x)) / h


def differentiate(
    h0: float, f: Callable[[float], float], x: float
) -> Stream[float]:
    """刻み幅 h を半分にしていく差分商の無限列。"""
    return Stream.iterate(lambda h: h / 2, h0).map(lambda h: easydiff(f, x, h))


def elimerror(n: int, s: Stream[float]) -> Stream[float]:
    """誤差項 O(h^n) を隣り合う 2 項から消去して精度を上げる。"""
    factor = 2.0**n
    return s.pairwise().map(lambda ab: (ab[1] * factor - ab[0]) / (factor - 1.0))


def order(s: Stream[float]) -> int:
    """先頭 3 項から誤差項の次数を推定する。"""
    a, b, c = s.take(3).to_list()
    return round(math.log2((a - c) / (b - c) - 1.0))


def improve(s: Stream[float]) -> Stream[float]:
    """列の次数を推定し、その誤差項を消去した改良列を返す。"""
    return elimerror(order(s), s)


def better_derivative(
    f: Callable[[float], float], x: float, *, h0: float = 1.0, eps: float = 1e-9
) -> float:
    return improve(differentiate(h0, f, x)).within(eps)


# --------------------------------------------------------------------------
# 3. 数値積分
#    easyintegrate → 区間を再帰的に二分して改良していく無限列 → within で打ち切り
# --------------------------------------------------------------------------


def integrate(
    f: Callable[[float], float], a: float, b: float
) -> Stream[float]:
    """台形近似から出発し、区間二分で精度を上げていく積分推定の無限列。"""

    def integ(a: float, b: float, fa: float, fb: float) -> Iterator[float]:
        m = (a + b) / 2
        fm = f(m)
        yield (fa + fb) * (b - a) / 2
        left = integ(a, m, fa, fm)
        right = integ(m, b, fm, fb)
        for lv, rv in zip(left, right):
            yield lv + rv

    return Stream(lambda: integ(a, b, f(a), f(b)))


def better_integral(
    f: Callable[[float], float], a: float, b: float, *, eps: float = 1e-9
) -> float:
    return integrate(f, a, b).within(eps)


def main() -> None:
    print("== ニュートン法の平方根 ==")
    for n in (2.0, 100.0, 1e6):
        got = newton_sqrt(n)
        print(f"  sqrt({n:g}) = {got:.12f}  (誤差 {abs(got - math.sqrt(n)):.2e})")

    print("\n== 数値微分 ==")
    # d/dx sin x = cos x
    got = better_derivative(math.sin, 1.0)
    print(f"  d/dx sin(1) = {got:.12f}  (cos 1 = {math.cos(1.0):.12f})")
    # d/dx exp x = exp x
    got = better_derivative(math.exp, 0.5)
    print(f"  d/dx exp(0.5) = {got:.12f}  (exp 0.5 = {math.exp(0.5):.12f})")

    print("\n== 数値積分 ==")
    # ∫_0^π sin x dx = 2
    got = better_integral(math.sin, 0.0, math.pi)
    print(f"  ∫_0^π sin x dx = {got:.12f}  (真値 2)")
    # ∫_0^1 4/(1+x^2) dx = π
    got = better_integral(lambda x: 4.0 / (1.0 + x * x), 0.0, 1.0)
    print(f"  ∫_0^1 4/(1+x^2) dx = {got:.12f}  (真値 π = {math.pi:.12f})")


if __name__ == "__main__":
    main()
