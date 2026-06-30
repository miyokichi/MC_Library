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

import pickle
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
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
from .rng import Generator, spawn_generators

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


def _chunk_sizes(n_trials: int, chunk_size: int) -> list[int]:
    full, remainder = divmod(n_trials, chunk_size)
    sizes = [chunk_size] * full
    if remainder:
        sizes.append(remainder)
    return sizes


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

        accumulators: dict[str, ColumnAccumulator] = {
            p.name: empty_accumulator() for p in self._plans
        }
        for chunk_acc in chunk_results:
            for name, acc in chunk_acc.items():
                accumulators[name] = accumulators[name].merge(acc)

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
