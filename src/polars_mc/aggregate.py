"""Chunk-mergeable statistics.

Across chunks we can only combine statistics that are *additive*.  We track,
per column, the standard set of moments (count / mean / M2 / min / max) and
merge them with Chan's parallel algorithm, which is numerically stable.

Quantiles are **not** additive.  For them we keep a bounded uniform random
sample per column using *bottom-k* reservoir sampling: every value is assigned a
random key and we keep the ``capacity`` items with the smallest keys.  This has
two nice properties:

* it is **order-independent and mergeable** -- merging two reservoirs is just
  keeping the bottom-k keys of the union, so the result never depends on how the
  stream was chunked; and
* when the whole stream fits in ``capacity`` every value is retained, so the
  quantile is **exact**.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
    "StatSpec",
    "OutputsSpec",
    "StatRequest",
    "ColumnPlan",
    "parse_outputs",
    "Moments",
    "Reservoir",
    "ColumnAccumulator",
    "empty_accumulator",
    "finalize_column",
]

# Default reservoir capacity for quantile estimation.
DEFAULT_QUANTILE_SAMPLE_SIZE = 200_000

# A single statistic: either a name ("mean", "std", ...) or ("q", probability).
StatSpec = str | tuple[str, float]
OutputsSpec = dict[str, list[StatSpec]]

# Statistics that are derived purely from the tracked moments.
_MOMENT_STATS = frozenset({"count", "sum", "mean", "var", "std", "min", "max"})


@dataclass(frozen=True)
class StatRequest:
    """A single requested statistic for one column, with its display label."""

    label: str
    kind: str  # "moment" | "quantile"
    moment: str | None = None
    prob: float | None = None


@dataclass(frozen=True)
class ColumnPlan:
    """All statistics requested for one output column."""

    name: str
    requests: tuple[StatRequest, ...]

    @property
    def quantile_probs(self) -> tuple[float, ...]:
        return tuple(r.prob for r in self.requests if r.prob is not None)

    @property
    def needs_sample(self) -> bool:
        return any(r.kind == "quantile" for r in self.requests)


def _parse_stat(stat: StatSpec) -> StatRequest:
    if isinstance(stat, tuple):
        kind, prob = stat
        if kind not in ("q", "quantile"):
            raise ValueError(f"Unknown tuple statistic: {stat!r}")
        if not (0.0 <= prob <= 1.0):
            raise ValueError(f"Quantile probability must be in [0, 1], got {prob}")
        return StatRequest(label=f"q{prob:g}", kind="quantile", prob=float(prob))

    if stat == "median":
        return StatRequest(label="median", kind="quantile", prob=0.5)
    if stat in _MOMENT_STATS:
        return StatRequest(label=stat, kind="moment", moment=stat)
    raise ValueError(
        f"Unknown statistic {stat!r}. Valid: {sorted(_MOMENT_STATS)}, "
        f"'median', or ('q', probability)."
    )


def parse_outputs(outputs: OutputsSpec) -> tuple[ColumnPlan, ...]:
    """Validate and normalise the user ``outputs`` spec into column plans."""
    if not outputs:
        raise ValueError("outputs must request at least one column statistic.")
    plans: list[ColumnPlan] = []
    for name, stats in outputs.items():
        if not stats:
            raise ValueError(f"Column {name!r} has no statistics requested.")
        requests = tuple(_parse_stat(s) for s in stats)
        plans.append(ColumnPlan(name=name, requests=requests))
    return tuple(plans)


@dataclass(frozen=True)
class Moments:
    """Additive moments for a single column.

    ``m2`` is the sum of squared deviations from the mean (so population
    variance is ``m2 / count`` and sample variance is ``m2 / (count - 1)``).
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def merge(self, other: Moments) -> Moments:
        """Combine two independent moment summaries (Chan et al., 1979)."""
        if self.count == 0:
            return other
        if other.count == 0:
            return self

        total = self.count + other.count
        delta = other.mean - self.mean
        mean = self.mean + delta * other.count / total
        m2 = self.m2 + other.m2 + delta * delta * self.count * other.count / total
        return Moments(
            count=total,
            mean=mean,
            m2=m2,
            minimum=min(self.minimum, other.minimum),
            maximum=max(self.maximum, other.maximum),
        )

    @property
    def sum(self) -> float:
        return self.mean * self.count

    @property
    def variance(self) -> float:
        if self.count < 2:
            return math.nan
        return self.m2 / (self.count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


@dataclass(frozen=True)
class Reservoir:
    """A bottom-k uniform sample of a stream, mergeable and order-independent."""

    capacity: int
    keys: NDArray[np.float64]
    values: NDArray[np.float64]
    count: int

    @classmethod
    def empty(cls, capacity: int) -> Reservoir:
        return cls(
            capacity=capacity,
            keys=np.empty(0, dtype=np.float64),
            values=np.empty(0, dtype=np.float64),
            count=0,
        )

    @classmethod
    def from_values(
        cls,
        values: ArrayLike,
        capacity: int,
        rng: np.random.Generator,
    ) -> Reservoir:
        vals = np.ascontiguousarray(values, dtype=np.float64)
        n = vals.size
        keys = rng.random(n)
        if n > capacity:
            idx = np.argpartition(keys, capacity)[:capacity]
            keys, vals = keys[idx].copy(), vals[idx].copy()
        return cls(capacity=capacity, keys=keys, values=vals, count=n)

    def merge(self, other: Reservoir) -> Reservoir:
        capacity = min(self.capacity, other.capacity)
        keys = np.concatenate([self.keys, other.keys])
        values = np.concatenate([self.values, other.values])
        if keys.size > capacity:
            idx = np.argpartition(keys, capacity)[:capacity]
            keys, values = keys[idx], values[idx]
        return Reservoir(
            capacity=capacity,
            keys=keys,
            values=values,
            count=self.count + other.count,
        )

    @property
    def is_exact(self) -> bool:
        """True if every observed value was retained (quantiles are exact)."""
        return self.count <= self.capacity

    def quantiles(self, probs: tuple[float, ...]) -> dict[float, float]:
        if self.values.size == 0:
            return {p: math.nan for p in probs}
        estimates = np.quantile(self.values, probs)
        return {p: float(q) for p, q in zip(probs, estimates)}


@dataclass
class ColumnAccumulator:
    """Running accumulator for one column: moments plus an optional reservoir."""

    moments: Moments = field(default_factory=Moments)
    reservoir: Reservoir | None = None

    def merge(self, other: ColumnAccumulator) -> ColumnAccumulator:
        if self.reservoir is None:
            reservoir = other.reservoir
        elif other.reservoir is None:
            reservoir = self.reservoir
        else:
            reservoir = self.reservoir.merge(other.reservoir)
        return ColumnAccumulator(
            moments=self.moments.merge(other.moments),
            reservoir=reservoir,
        )


def empty_accumulator(plan: ColumnPlan, capacity: int) -> ColumnAccumulator:
    reservoir = Reservoir.empty(capacity) if plan.needs_sample else None
    return ColumnAccumulator(moments=Moments(), reservoir=reservoir)


def finalize_column(plan: ColumnPlan, acc: ColumnAccumulator) -> dict[str, float]:
    """Produce the final ``{label: value}`` mapping for one column."""
    m = acc.moments
    moment_values = {
        "count": float(m.count),
        "sum": m.sum,
        "mean": m.mean if m.count else math.nan,
        "var": m.variance,
        "std": m.std,
        "min": m.minimum if m.count else math.nan,
        "max": m.maximum if m.count else math.nan,
    }

    probs = plan.quantile_probs
    quantile_values: dict[float, float] = {}
    if probs:
        reservoir = acc.reservoir or Reservoir.empty(0)
        quantile_values = reservoir.quantiles(probs)

    result: dict[str, float] = {}
    for req in plan.requests:
        if req.kind == "moment":
            assert req.moment is not None
            result[req.label] = moment_values[req.moment]
        else:
            assert req.prob is not None
            result[req.label] = quantile_values[req.prob]
    return result
