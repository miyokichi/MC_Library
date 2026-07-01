"""The high-level ``Simulation`` engine.

This is now a thin wrapper that composes the standalone building blocks:

* :mod:`polars_mc.sampling` -- reproducible random generation,
* :mod:`polars_mc.chunk` -- apply the trial to a chunk,
* :mod:`polars_mc.reduce` / :mod:`polars_mc.aggregate` -- mergeable statistics,
* :mod:`polars_mc.chunking` -- the generic chunked map-reduce (and backends).

Each of those can be used directly; ``Simulation`` just wires them together for
the common "sample -> trial -> aggregate" case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .aggregate import (
    DEFAULT_QUANTILE_SAMPLE_SIZE,
    ColumnAccumulator,
    ColumnPlan,
    OutputsSpec,
    finalize_column,
    parse_outputs,
)
from .chunk import TrialFn, TrialStyle, classify_trial, run_chunk
from .chunking import DEFAULT_CHUNK_SIZE, Backend, map_reduce_chunks
from .distributions import Distribution
from .rng import Generator
from .result import SimulationResult

__all__ = [
    "Simulation",
    "Backend",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
]

ChunkAccumulators = dict[str, ColumnAccumulator]


@dataclass(frozen=True)
class _TrialMapper:
    """Picklable ``map_fn`` for one chunk: sample -> trial -> accumulate."""

    inputs: dict[str, Distribution]
    trial: TrialFn
    plans: tuple[ColumnPlan, ...]
    capacity: int
    style: TrialStyle

    def __call__(self, n: int, rng: Generator) -> ChunkAccumulators:
        return run_chunk(
            inputs=self.inputs,
            trial=self.trial,
            plans=self.plans,
            n=n,
            rng=rng,
            capacity=self.capacity,
            style=self.style,
        )


def _merge_accumulators(
    left: ChunkAccumulators,
    right: ChunkAccumulators,
) -> ChunkAccumulators:
    return {name: left[name].merge(right[name]) for name in left}


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
        Reservoir capacity for quantile estimation.  If the total number of
        trials does not exceed this, quantiles are exact.
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
        backend: Backend = "sequential",
        n_workers: int | None = None,
    ) -> SimulationResult:
        """Run ``n_trials`` trials and return the aggregated result.

        Parameters
        ----------
        chunk_size:
            Rows per chunk.  Defaults to ``min(n_trials, DEFAULT_CHUNK_SIZE)``.
        seed:
            Master seed.  Identical seed + chunk size always yields identical
            results, regardless of ``backend`` or ``n_workers``.
        backend:
            ``"sequential"`` (default) or ``"processes"``.  The latter
            distributes whole chunks across a process pool -- useful for
            CPU-heavy trials (e.g. Shapely) that Polars threads cannot
            parallelise.  Requires a picklable ``trial`` (module-level function).
        n_workers:
            Worker process count for ``backend="processes"`` (default: all CPUs).
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

        effective_chunk = min(n_trials, chunk_size or DEFAULT_CHUNK_SIZE)
        mapper = _TrialMapper(
            inputs=self.inputs,
            trial=self.trial,
            plans=self._plans,
            capacity=self.quantile_sample_size,
            style=self._style,
        )
        accumulators = map_reduce_chunks(
            n_trials,
            chunk_size=effective_chunk,
            seed=seed,
            map_fn=mapper,
            reduce_fn=_merge_accumulators,
            backend=backend,
            n_workers=n_workers,
        )

        stats = {
            plan.name: finalize_column(plan, accumulators[plan.name])
            for plan in self._plans
        }
        approximate = frozenset(
            p.name
            for p in self._plans
            if p.needs_sample
            and accumulators[p.name].reservoir is not None
            and not accumulators[p.name].reservoir.is_exact  # type: ignore[union-attr]
        )
        n_chunks = len(range(0, n_trials, effective_chunk))

        return SimulationResult(
            n=n_trials,
            stats=stats,
            approximate_columns=approximate,
            n_chunks=n_chunks,
            seed=seed,
        )
