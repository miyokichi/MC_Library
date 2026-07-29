"""Tests for the lazy Stream combinator DSL and the article's numeric examples."""

from __future__ import annotations

import math

import pytest

from mc_lib.stream import Stream


# -- basic combinators ------------------------------------------------------


def test_iterate_and_take() -> None:
    powers = Stream.iterate(lambda x: x * 2, 1).take(5).to_list()
    assert powers == [1, 2, 4, 8, 16]


def test_iterate_is_rerunnable() -> None:
    s = Stream.iterate(lambda x: x + 1, 0).take(3)
    assert s.to_list() == [0, 1, 2]
    # Consuming again restarts from the beginning (streams are values).
    assert s.to_list() == [0, 1, 2]


def test_map_and_filter() -> None:
    evens = (
        Stream.from_iterable(range(10))
        .map(lambda x: x * x)
        .filter(lambda x: x % 2 == 0)
        .to_list()
    )
    assert evens == [0, 4, 16, 36, 64]


def test_scan_emits_init_then_running_fold() -> None:
    sums = Stream.from_iterable([1, 2, 3, 4]).scan(lambda a, x: a + x, 0).to_list()
    assert sums == [0, 1, 3, 6, 10]


def test_reduce_is_left_fold() -> None:
    total = Stream.from_iterable([1, 2, 3, 4]).reduce(lambda a, x: a + x, 0)
    assert total == 10


def test_take_while_and_take_until() -> None:
    s = Stream.from_iterable([1, 2, 3, 4, 1])
    assert s.take_while(lambda x: x < 3).to_list() == [1, 2]
    # take_until includes the first element that satisfies the predicate.
    assert s.take_until(lambda x: x >= 3).to_list() == [1, 2, 3]


def test_pairwise() -> None:
    pairs = Stream.from_iterable([1, 2, 3, 4]).pairwise().to_list()
    assert pairs == [(1, 2), (2, 3), (3, 4)]


def test_pairwise_empty_and_singleton() -> None:
    assert Stream.from_iterable([]).pairwise().to_list() == []
    assert Stream.from_iterable([7]).pairwise().to_list() == []


def test_zip_with() -> None:
    a = Stream.from_iterable([1, 2, 3])
    b = Stream.from_iterable([10, 20, 30, 40])
    assert a.zip_with(b, lambda x, y: x + y).to_list() == [11, 22, 33]


def test_drop_and_first() -> None:
    s = Stream.iterate(lambda x: x + 1, 0)
    assert s.drop(3).first() == 3
    assert s.take(0).to_list() == []


# -- convergence combinators (the article's within / relative) --------------


def test_within_on_constant_stream() -> None:
    assert Stream.from_iterable([1.0, 1.0, 1.0]).within(0.0) == 1.0


def test_within_raises_if_never_converges() -> None:
    with pytest.raises(ValueError):
        Stream.from_iterable([1.0, 2.0, 3.0]).within(1e-9)


def test_relative_tolerance() -> None:
    # First pair differs by 3 (> 1% of 103); second pair by 0.5 (< 1% of 103.5).
    s = Stream.from_iterable([100.0, 103.0, 103.5])
    assert s.relative(0.01) == 103.5


# -- article example 1: Newton-Raphson square root --------------------------


@pytest.mark.parametrize("n", [2.0, 100.0, 1e6, 0.25])
def test_newton_sqrt_matches_math_sqrt(n: float) -> None:
    from examples.whyfp_numeric import newton_sqrt

    assert newton_sqrt(n) == pytest.approx(math.sqrt(n), rel=1e-9)


# -- article example 2: numerical differentiation ---------------------------


def test_better_derivative_of_sin_is_cos() -> None:
    from examples.whyfp_numeric import better_derivative

    assert better_derivative(math.sin, 1.0) == pytest.approx(math.cos(1.0), abs=1e-6)


def test_better_derivative_of_exp() -> None:
    from examples.whyfp_numeric import better_derivative

    assert better_derivative(math.exp, 0.5) == pytest.approx(math.exp(0.5), abs=1e-6)


# -- article example 3: numerical integration -------------------------------


def test_integrate_sin_over_0_pi_is_2() -> None:
    from examples.whyfp_numeric import better_integral

    assert better_integral(math.sin, 0.0, math.pi) == pytest.approx(2.0, abs=1e-7)


def test_integrate_gives_pi() -> None:
    from examples.whyfp_numeric import better_integral

    got = better_integral(lambda x: 4.0 / (1.0 + x * x), 0.0, 1.0)
    assert got == pytest.approx(math.pi, abs=1e-7)
