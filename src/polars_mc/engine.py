"""The simulation engine: chunk planning, execution and merging.

By default parallelism is delegated to Polars' internal thread pool, which
parallelises the expression evaluation *within* each chunk while the chunk loop
runs sequentially -- keeping memory bounded and results perfectly reproducible
for a given seed.

For CPU-heavy trials whose work is *not* Polars expressions (e.g. NumPy-object
work such as Shapely geometry operations), Polars threads do not help.  There
``backend="processes"`` distributes whole chunks across a process pool.  Because
each chunk has its own independent, pre-seeded RNG, the parallel result is
identical to the sequential one, bit for bit.
"""

from __future__ import annotations

import math
import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from itertools import count
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
from .estimate import DEFAULT_MAX_TRIALS
from .result import SimulationResult
from .rng import Generator, generator_stream, spawn_generators
from .sampling import DEFAULT_STREAM_CHUNK
from .stream import Stream

__all__ = [
    "Simulation",
    "Backend",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
]

DEFAULT_CHUNK_SIZE = 250_000
DEFAULT_QUANTILE_SAMPLE_SIZE = 200_000

Backend = Literal["sequential", "processes"]

# One unit of work handed to the executor: (chunk size, rng, quantile sample size).
ChunkTask = tuple[int, Generator, int]

# Per-column accumulators for one chunk, keyed by column name.
ChunkAccumulators = dict[str, ColumnAccumulator]


def _chunk_sizes(n_trials: int, chunk_size: int) -> list[int]:
    full, remainder = divmod(n_trials, chunk_size)
    sizes = [chunk_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes


def _merge_all(
    acc: ChunkAccumulators, chunk: ChunkAccumulators
) -> ChunkAccumulators:
    """Monoidal combine of two per-column accumulator maps (used as a fold step)."""
    return {name: acc[name].merge(chunk[name]) for name in acc}


@dataclass(frozen=True)
class _WorkerContext:
    """The per-run configuration shared by every chunk in a process pool."""

    inputs: dict[str, Distribution]
    trial: TrialFn
    plans: tuple[ColumnPlan, ...]
    style: TrialStyle


# Set once per worker process via the pool initializer, so the (potentially
# larger) shared context is pickled once per worker rather than once per chunk.
_WORKER_CONTEXT: _WorkerContext | None = None


def _init_worker(context: _WorkerContext) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = context


def _execute_task(task: ChunkTask) -> dict[str, ColumnAccumulator]:
    assert _WORKER_CONTEXT is not None, "worker context was not initialised"
    ctx = _WORKER_CONTEXT
    n, rng, sample_size = task
    return run_chunk(
        inputs=ctx.inputs,
        trial=ctx.trial,
        plans=ctx.plans,
        n=n,
        rng=rng,
        sample_size=sample_size,
        style=ctx.style,
    )


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
            ``"sequential"`` (default) runs chunks in order, relying on Polars'
            internal threads within each chunk.  ``"processes"`` distributes
            whole chunks across a process pool -- useful for CPU-heavy trials
            (e.g. Shapely) that Polars threads cannot parallelise.  Requires a
            picklable ``trial`` (define it at module level, not as a lambda).
        n_workers:
            Worker process count for ``backend="processes"`` (default: all CPUs).
        """
        if n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {n_trials}.")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        if n_workers is not None and n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}.")

        effective_chunk = min(n_trials, chunk_size or DEFAULT_CHUNK_SIZE)
        sizes = _chunk_sizes(n_trials, effective_chunk)
        generators = spawn_generators(seed, len(sizes))

        needs_sample = any(p.needs_sample for p in self._plans)
        exhaustive_sample = n_trials <= self.quantile_sample_size

        tasks: list[ChunkTask] = [
            (
                chunk_n,
                rng,
                self._sample_size_for(
                    chunk_n, n_trials, needs_sample, exhaustive_sample
                ),
            )
            for chunk_n, rng in zip(sizes, generators)
        ]

        # A single chunk never benefits from a process pool's overhead.
        use_processes = backend == "processes" and len(tasks) > 1
        if backend == "processes":
            self._check_picklable_trial()
        if use_processes:
            chunk_results = self._run_processes(tasks, n_workers)
        elif backend in ("sequential", "processes"):
            chunk_results = (_run_task_locally(self, task) for task in tasks)
        else:
            raise ValueError(f"Unknown backend {backend!r}.")

        # Fold the per-chunk accumulators with the same monoid the streaming
        # path uses -- batch is just this fold over a bounded chunk stream.
        accumulators = Stream.from_iterable(chunk_results).reduce(
            _merge_all,
            {p.name: empty_accumulator() for p in self._plans},
        )

        stats = {
            plan.name: finalize_column(plan, accumulators[plan.name])
            for plan in self._plans
        }
        approximate = frozenset(
            p.name for p in self._plans if p.needs_sample and not exhaustive_sample
        )

        return SimulationResult(
            n_trials=n_trials,
            n_chunks=len(sizes),
            seed=seed,
            stats=stats,
            approximate_columns=approximate,
        )

    def run_until(
        self,
        target: str,
        *,
        target_se: float | None = None,
        within: float | None = None,
        relative: float | None = None,
        seed: int = 0,
        chunk_size: int | None = None,
        max_trials: int = DEFAULT_MAX_TRIALS,
    ) -> SimulationResult:
        """Run adaptively until ``target``'s running mean converges.

        Instead of a fixed ``n_trials``, this consumes an *unbounded* stream of
        chunks and stops as soon as the ``target`` output column's estimate is
        good enough -- the article's "generate approximations, stop when close"
        skeleton applied to the whole simulation.  All requested outputs are
        finalised at the stopping point.

        Parameters
        ----------
        target:
            Name of the output column whose mean drives the stopping decision.
        target_se:
            Stop when ``target``'s standard error of the mean is ``<= target_se``.
        within / relative:
            Stop when successive running means differ by ``<= within`` (absolute)
            or ``<= relative * |mean|``.  These mirror the article's ``within`` /
            ``relative`` convergence tests.
        seed:
            Master seed; reproducible exactly like :meth:`run`.
        chunk_size:
            Samples per chunk (defaults to a stream-sized chunk so convergence is
            checked often).
        max_trials:
            Hard cap; if reached first, the result is flagged ``converged=False``.

        Notes
        -----
        Because every chunk is i.i.d., the first ``quantile_sample_size`` retained
        samples form a valid uniform sample, so quantiles remain sound (flagged
        approximate once the run outgrows that budget).
        """
        names = {p.name for p in self._plans}
        if target not in names:
            raise ValueError(
                f"target {target!r} is not an output column; choose from {sorted(names)}."
            )
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        if max_trials < 1:
            raise ValueError(f"max_trials must be >= 1, got {max_trials}.")

        chunk = min(max_trials, chunk_size or DEFAULT_STREAM_CHUNK)
        needs_sample = any(p.needs_sample for p in self._plans)

        def execute(index: int, rng: Generator) -> ChunkAccumulators:
            return run_chunk(
                inputs=self.inputs,
                trial=self.trial,
                plans=self._plans,
                n=chunk,
                rng=rng,
                sample_size=self._stream_sample_size(index, chunk, needs_sample),
                style=self._style,
            )

        # The unbounded chunk stream is the shared generator stream, executed:
        # same seeding as `run`, just consumed lazily instead of by the chunk.
        chunks = Stream(count).zip_with(generator_stream(seed), execute)
        empty: ChunkAccumulators = {p.name: empty_accumulator() for p in self._plans}
        running = chunks.scan(_merge_all, empty).drop(1)

        max_chunks = max(1, math.ceil(max_trials / chunk))
        prev_mean: float | None = None
        last = empty
        converged = False
        n_chunks_used = 0
        for i, acc in enumerate(running, start=1):
            last = acc
            n_chunks_used = i
            moments = acc[target].moments
            if (
                target_se is not None
                and moments.count >= 2
                and moments.standard_error <= target_se
            ):
                converged = True
                break
            if prev_mean is not None:
                delta = abs(moments.mean - prev_mean)
                if within is not None and delta <= within:
                    converged = True
                    break
                if relative is not None and delta <= relative * abs(moments.mean):
                    converged = True
                    break
            prev_mean = moments.mean
            if i >= max_chunks:
                break

        stats = {
            plan.name: finalize_column(plan, last[plan.name]) for plan in self._plans
        }
        n_used = last[target].moments.count
        exhaustive = not needs_sample or n_used <= self.quantile_sample_size
        approximate = frozenset(
            p.name for p in self._plans if p.needs_sample and not exhaustive
        )
        return SimulationResult(
            n_trials=n_used,
            n_chunks=n_chunks_used,
            seed=seed,
            stats=stats,
            approximate_columns=approximate,
            converged=converged,
            standard_error=last[target].moments.standard_error,
            target=target,
        )

    def _run_processes(
        self,
        tasks: list[ChunkTask],
        n_workers: int | None,
    ) -> Iterator[dict[str, ColumnAccumulator]]:
        context = _WorkerContext(
            inputs=self.inputs,
            trial=self.trial,
            plans=self._plans,
            style=self._style,
        )
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init_worker,
            initargs=(context,),
        ) as executor:
            # map preserves input order, so the merge is identical to sequential.
            yield from executor.map(_execute_task, tasks)

    def _check_picklable_trial(self) -> None:
        try:
            pickle.dumps(self.trial)
        except Exception as exc:  # noqa: BLE001 - re-raised as a clear error
            raise ValueError(
                "backend='processes' requires a picklable trial. Define it as a "
                "module-level function (not a lambda or a locally-defined "
                "closure), and guard your script entry point with "
                "`if __name__ == '__main__':`."
            ) from exc

    def _stream_sample_size(self, index: int, chunk: int, needs_sample: bool) -> int:
        """Rows chunk ``index`` of an unbounded run retains for quantiles.

        Chunks are i.i.d., so filling the budget from the leading chunks leaves
        the pooled sample uniform.  After ``index`` chunks the pool holds
        ``min(index * chunk, budget)`` rows, so each chunk's share is a pure
        function of its index -- no counter to carry, which keeps the chunk
        stream re-runnable.
        """
        if not needs_sample:
            return 0
        remaining = self.quantile_sample_size - index * chunk
        return max(0, min(chunk, remaining))

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


def _run_task_locally(
    sim: Simulation,
    task: ChunkTask,
) -> dict[str, ColumnAccumulator]:
    n, rng, sample_size = task
    return run_chunk(
        inputs=sim.inputs,
        trial=sim.trial,
        plans=sim._plans,
        n=n,
        rng=rng,
        sample_size=sample_size,
        style=sim._style,
    )
