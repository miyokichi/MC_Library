"""Standalone statistics: chunk-mergeable summaries of your own data.

Decoupled from sampling and trials -- feed in any Polars frame or mapping of
arrays.  Two styles are provided:

* **functional** -- :func:`summarize_partial` returns a mergeable :class:`Stats`
  and :func:`merge` folds several together; :func:`summarize` is the one-shot
  shortcut.
* **stateful** -- :class:`Summarizer` accumulates chunks via ``update`` and is
  handy inside your own loop.

All statistics are the same ones the simulation uses (mean/std/min/max/quantiles
etc.), combined exactly across chunks (quantiles via a bounded reservoir).
"""

from __future__ import annotations

import numpy as np

from .aggregate import (
    DEFAULT_QUANTILE_SAMPLE_SIZE,
    ColumnAccumulator,
    ColumnPlan,
    OutputsSpec,
    empty_accumulator,
    finalize_column,
    parse_outputs,
)
from .reduce import DataLike, accumulate_columns, to_dataframe
from .result import StatResult

__all__ = ["Stats", "summarize_partial", "merge", "summarize", "Summarizer"]


def _finalize(
    plans: tuple[ColumnPlan, ...],
    accumulators: dict[str, ColumnAccumulator],
) -> StatResult:
    stats = {p.name: finalize_column(p, accumulators[p.name]) for p in plans}
    n = max((acc.moments.count for acc in accumulators.values()), default=0)
    approximate = frozenset(
        p.name
        for p in plans
        if p.needs_sample
        and accumulators[p.name].reservoir is not None
        and not accumulators[p.name].reservoir.is_exact  # type: ignore[union-attr]
    )
    return StatResult(n=n, stats=stats, approximate_columns=approximate)


class Stats:
    """A mergeable, order-independent container of partial statistics."""

    def __init__(
        self,
        plans: tuple[ColumnPlan, ...],
        accumulators: dict[str, ColumnAccumulator],
        capacity: int,
    ) -> None:
        self._plans = plans
        self._accumulators = accumulators
        self._capacity = capacity

    def merge(self, other: Stats) -> Stats:
        if self._plans != other._plans:
            raise ValueError("cannot merge Stats built from different outputs specs.")
        merged = {
            name: self._accumulators[name].merge(other._accumulators[name])
            for name in self._accumulators
        }
        return Stats(self._plans, merged, min(self._capacity, other._capacity))

    def result(self) -> StatResult:
        return _finalize(self._plans, self._accumulators)


def summarize_partial(
    data: DataLike,
    outputs: OutputsSpec,
    *,
    quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE,
    rng: np.random.Generator | None = None,
) -> Stats:
    """Reduce ``data`` to a mergeable :class:`Stats` (low-level, functional).

    ``rng`` seeds the quantile reservoir; pass distinct generators (or use
    :class:`Summarizer`) when combining several partials, so their samples are
    independent.
    """
    plans = parse_outputs(outputs)
    df = to_dataframe(data)
    generator = rng if rng is not None else np.random.default_rng()
    accumulators = accumulate_columns(df, plans, generator, quantile_sample_size)
    return Stats(plans, accumulators, quantile_sample_size)


def merge(*stats: Stats) -> Stats:
    """Fold several :class:`Stats` into one."""
    if not stats:
        raise ValueError("merge requires at least one Stats.")
    combined = stats[0]
    for partial in stats[1:]:
        combined = combined.merge(partial)
    return combined


def summarize(
    data: DataLike,
    outputs: OutputsSpec,
    *,
    quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE,
    rng: np.random.Generator | None = None,
) -> StatResult:
    """One-shot statistics over ``data`` (the common case)."""
    return summarize_partial(
        data, outputs, quantile_sample_size=quantile_sample_size, rng=rng
    ).result()


class Summarizer:
    """Stateful accumulator: feed chunks with :meth:`update`, read :meth:`result`.

    Each :meth:`update` draws fresh reservoir keys from an internal, seeded
    generator, so results are reproducible and independent across chunks.
    """

    def __init__(
        self,
        outputs: OutputsSpec,
        *,
        quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE,
        seed: int = 0,
    ) -> None:
        self._plans = parse_outputs(outputs)
        self._capacity = quantile_sample_size
        self._rng = np.random.default_rng(seed)
        self._accumulators = {
            p.name: empty_accumulator(p, quantile_sample_size) for p in self._plans
        }

    def update(self, data: DataLike) -> Summarizer:
        df = to_dataframe(data)
        chunk = accumulate_columns(df, self._plans, self._rng, self._capacity)
        for name in self._accumulators:
            self._accumulators[name] = self._accumulators[name].merge(chunk[name])
        return self

    def merge(self, other: Summarizer | Stats) -> Summarizer:
        stats = other.stats() if isinstance(other, Summarizer) else other
        combined = self.stats().merge(stats)
        self._accumulators = combined._accumulators
        return self

    def stats(self) -> Stats:
        return Stats(self._plans, dict(self._accumulators), self._capacity)

    def result(self) -> StatResult:
        return _finalize(self._plans, self._accumulators)
