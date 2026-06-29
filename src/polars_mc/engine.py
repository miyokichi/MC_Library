"""The simulation engine: chunk planning, execution and merging.

Parallelism is delegated entirely to Polars' internal thread pool, which
parallelises the expression evaluation *within* each chunk.  The chunk loop
itself is sequential, which keeps memory bounded and results perfectly
reproducible for a given seed.  Heavier backends (multiprocessing, Dask) can be
layered on later without changing the user-facing API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .aggregate import (
    ColumnAccumulator,
    ColumnPlan,
    OutputsSpec,
    empty_accumulator,
    finalize_column,
    parse_outputs,
)
from .chunk import TrialFn, TrialStyle, classify_trial, run_chunk
from .distributions import Distribution
from .result import SimulationResult
from .rng import spawn_generators

__all__ = ["Simulation", "DEFAULT_CHUNK_SIZE", "DEFAULT_QUANTILE_SAMPLE_SIZE"]

DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_QUANTILE_SAMPLE_SIZE = 200_000


def _chunk_sizes(n_trials: int, chunk_size: int) -> list[int]:
    full, remainder = divmod(n_trials, chunk_size)
    sizes = [chunk_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes


@dataclass
class Simulation:
    """A reusable Monte Carlo simulation definition.

    Parameters
    ----------
    inputs:
        Mapping of input column name to its :class:`Distribution`.
    trial:
        The vectorized trial, in one of two styles (auto-detected from its
        signature, or forced with ``trial_style``):

        * **array** -- an ordinary function whose parameters are input names; it
          receives NumPy arrays (one per chunk) and returns
          ``{output_name: array}``.  NumPy broadcasting makes single-trial-style
          code run vectorized over the whole chunk.
        * **frame** -- a function ``(pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame``
          built from Polars expressions.

        Either way it must produce every column named in ``outputs``.
    outputs:
        Mapping of output column name to the list of statistics to compute.
        Statistics: ``"count"``, ``"sum"``, ``"mean"``, ``"var"``, ``"std"``,
        ``"min"``, ``"max"``, ``"median"`` or ``("q", probability)``.
    quantile_sample_size:
        Target size of the pooled subsample used to estimate quantiles.  If the
        total number of trials does not exceed this budget, quantiles are exact.
    trial_style:
        ``"auto"`` (default), ``"array"`` or ``"frame"``.
    """

    inputs: dict[str, Distribution]
    trial: TrialFn
    outputs: OutputsSpec
    quantile_sample_size: int = DEFAULT_QUANTILE_SAMPLE_SIZE
    trial_style: Literal["auto", "array", "frame"] = "auto"
    _plans: tuple[ColumnPlan, ...] = field(init=False, repr=False)
    _style: TrialStyle = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ValueError("inputs must contain at least one distribution.")
        if self.quantile_sample_size < 1:
            raise ValueError("quantile_sample_size must be >= 1.")
        self._plans = parse_outputs(self.outputs)
        if self.trial_style == "auto":
            self._style = classify_trial(self.trial, set(self.inputs))
        else:
            self._style = self.trial_style

    def run(
        self,
        n_trials: int,
        *,
        chunk_size: int | None = None,
        seed: int = 0,
    ) -> SimulationResult:
        """Run ``n_trials`` trials and return the aggregated result."""
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

        effective_chunk = min(n_trials, chunk_size or DEFAULT_CHUNK_SIZE)
        sizes = _chunk_sizes(n_trials, effective_chunk)
        generators = spawn_generators(seed, len(sizes))

        needs_sample = any(p.needs_sample for p in self._plans)
        exhaustive_sample = n_trials <= self.quantile_sample_size

        accumulators: dict[str, ColumnAccumulator] = {
            p.name: empty_accumulator() for p in self._plans
        }

        for chunk_n, rng in zip(sizes, generators):
            sample_size = self._sample_size_for(
                chunk_n, n_trials, needs_sample, exhaustive_sample
            )
            chunk_acc = run_chunk(
                inputs=self.inputs,
                trial=self.trial,
                plans=self._plans,
                n=chunk_n,
                rng=rng,
                sample_size=sample_size,
                style=self._style,
            )
            for name, acc in chunk_acc.items():
                accumulators[name] = accumulators[name].merge(acc)

        stats = {
            plan.name: finalize_column(plan, accumulators[plan.name])
            for plan in self._plans
        }
        approximate = frozenset(
            p.name
            for p in self._plans
            if p.needs_sample and not exhaustive_sample
        )

        return SimulationResult(
            n_trials=n_trials,
            n_chunks=len(sizes),
            seed=seed,
            stats=stats,
            approximate_columns=approximate,
        )

    def _sample_size_for(
        self,
        chunk_n: int,
        n_trials: int,
        needs_sample: bool,
        exhaustive: bool,
    ) -> int:
        if not needs_sample:
            return 0
        if exhaustive:
            return chunk_n
        proportional = round(self.quantile_sample_size * chunk_n / n_trials)
        return max(1, min(chunk_n, proportional))
