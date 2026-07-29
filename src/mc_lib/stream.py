"""Lazy stream combinators: the composable core (the "glue" of the library).

This module is the functional heart of ``mc_lib``, distilled from John
Hughes' *Why Functional Programming Matters*.  A :class:`Stream` is a
*re-runnable* lazy sequence -- a value, not a one-shot iterator -- so the same
stream can be assembled by composition and consumed more than once with
identical results.

The article identifies two kinds of "glue" for building programs out of small
parts, and both live here:

* **higher-order functions** -- :meth:`Stream.map`, :meth:`Stream.filter`,
  :meth:`Stream.scan`, :meth:`Stream.reduce`, ... compose big computations from
  small ones.
* **lazy infinite lists** -- :meth:`Stream.iterate` (the article's ``repeat``)
  generates an unbounded sequence of successive approximations; the convergence
  combinators :meth:`Stream.within` / :meth:`Stream.relative` consume just
  enough of it to reach an answer.

There is deliberately no NumPy or Polars here: this layer is pure Python and
mirrors the article directly.  Newton-Raphson square root, for instance, is
literally::

    Stream.iterate(lambda x: (x + n / x) / 2, x0).within(eps)
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import islice
from typing import Generic, TypeVar

T = TypeVar("T")
U = TypeVar("U")
S = TypeVar("S")

__all__ = ["Stream"]


class Stream(Generic[T]):
    """A re-runnable lazy sequence.

    Wraps a *thunk* ``() -> Iterator[T]`` so that iterating twice restarts the
    sequence from the beginning -- streams behave like values, just as lists do
    in the article.  Combinators return new streams and never consume the source
    eagerly; only the terminal operations (:meth:`reduce`, :meth:`within`,
    :meth:`relative`, :meth:`first`, :meth:`to_list`) drive iteration.
    """

    __slots__ = ("_make",)

    def __init__(self, make: Callable[[], Iterator[T]]) -> None:
        self._make = make

    def __iter__(self) -> Iterator[T]:
        return self._make()

    # -- constructors ----------------------------------------------------

    @classmethod
    def iterate(cls, f: Callable[[T], T], x0: T) -> Stream[T]:
        """Infinite stream ``x0, f(x0), f(f(x0)), ...`` (the article's ``repeat``)."""

        def gen() -> Iterator[T]:
            x = x0
            while True:
                yield x
                x = f(x)

        return cls(gen)

    @classmethod
    def from_iterable(cls, source: Iterable[T]) -> Stream[T]:
        """Stream over a re-iterable source (list, range, ...).

        Passing a one-shot iterator yields a single-use stream; wrap a factory
        with the constructor directly if you need repeatability from a generator.
        """
        return cls(lambda: iter(source))

    # -- lazy combinators (higher-order glue) ----------------------------

    def map(self, f: Callable[[T], U]) -> Stream[U]:
        """Apply ``f`` to every element, lazily."""
        return Stream(lambda: map(f, self._make()))

    def filter(self, predicate: Callable[[T], bool]) -> Stream[T]:
        """Keep only elements for which ``predicate`` is true, lazily."""
        return Stream(lambda: filter(predicate, self._make()))

    def scan(self, f: Callable[[S, T], S], init: S) -> Stream[S]:
        """Running fold (Haskell ``scanl``): emit ``init``, then each accumulation.

        The output is ``init, f(init, x0), f(f(init, x0), x1), ...`` -- one more
        element than the source, starting with ``init``.
        """

        def gen() -> Iterator[S]:
            acc = init
            yield acc
            for x in self._make():
                acc = f(acc, x)
                yield acc

        return Stream(gen)

    def take(self, n: int) -> Stream[T]:
        """First ``n`` elements (fewer if the source is shorter)."""
        return Stream(lambda: islice(self._make(), max(0, n)))

    def drop(self, n: int) -> Stream[T]:
        """All but the first ``n`` elements."""
        return Stream(lambda: islice(self._make(), max(0, n), None))

    def take_while(self, predicate: Callable[[T], bool]) -> Stream[T]:
        """Longest prefix whose elements all satisfy ``predicate``."""

        def gen() -> Iterator[T]:
            for x in self._make():
                if not predicate(x):
                    return
                yield x

        return Stream(gen)

    def take_until(self, stop: Callable[[T], bool]) -> Stream[T]:
        """Prefix up to and *including* the first element that satisfies ``stop``.

        This is the streaming analogue of the article's convergence check: keep
        pulling approximations until one is "good enough".
        """

        def gen() -> Iterator[T]:
            for x in self._make():
                yield x
                if stop(x):
                    return

        return Stream(gen)

    def pairwise(self) -> Stream[tuple[T, T]]:
        """Consecutive overlapping pairs ``(x0, x1), (x1, x2), ...``."""

        def gen() -> Iterator[tuple[T, T]]:
            it = self._make()
            try:
                prev = next(it)
            except StopIteration:
                return
            for cur in it:
                yield (prev, cur)
                prev = cur

        return Stream(gen)

    def zip_with(self, other: Stream[U], f: Callable[[T, U], S]) -> Stream[S]:
        """Combine two streams element-wise with ``f``; stops at the shorter."""
        return Stream(lambda: map(f, self._make(), other._make()))

    # -- terminal operations ---------------------------------------------

    def reduce(self, f: Callable[[S, T], S], init: S) -> S:
        """Left fold (the article's ``reduce`` / Haskell ``foldl``)."""
        acc = init
        for x in self._make():
            acc = f(acc, x)
        return acc

    def first(self) -> T:
        """The first element; raises :class:`StopIteration` if the stream is empty."""
        return next(self._make())

    def to_list(self) -> list[T]:
        """Materialise the (finite!) stream into a list."""
        return list(self._make())

    def within(self: Stream[float], eps: float) -> float:
        """First value whose successor differs by ``<= eps`` (article's ``within``).

        Consumes the stream of successive approximations until two consecutive
        values are within absolute tolerance ``eps`` of each other, then returns
        the later one.
        """
        for a, b in self.pairwise():
            if abs(a - b) <= eps:
                return b
        raise ValueError("stream exhausted before converging within eps")

    def relative(self: Stream[float], eps: float) -> float:
        """Like :meth:`within` but with a *relative* tolerance (article's ``relative``).

        Stops when ``abs(a - b) <= eps * abs(b)``.
        """
        for a, b in self.pairwise():
            if abs(a - b) <= eps * abs(b):
                return b
        raise ValueError("stream exhausted before converging relative to eps")
