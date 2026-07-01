"""Generic chunked map-reduce with reproducible per-chunk RNG.

This is the bare execution engine, independent of trials or statistics:

* split ``n`` items into chunks,
* give each chunk its own independent, pre-seeded ``numpy`` generator,
* ``map_fn(chunk_n, rng)`` produces a partial result, and
* ``reduce_fn`` folds the partials into one.

Because each chunk is independently seeded, the result is identical whether the
chunks run sequentially or across a process pool.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from typing import Literal, TypeVar

from .rng import Generator, spawn_generators

__all__ = ["Backend", "DEFAULT_CHUNK_SIZE", "chunk_sizes", "map_reduce_chunks"]

DEFAULT_CHUNK_SIZE = 250_000

T = TypeVar("T")
Backend = Literal["sequential", "processes"]

MapFn = Callable[[int, Generator], T]
ReduceFn = Callable[[T, T], T]

ChunkTask = tuple[int, Generator]


def chunk_sizes(n: int, chunk_size: int) -> list[int]:
    """Split ``n`` items into chunks of at most ``chunk_size``."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
    effective = min(n, chunk_size)
    full, remainder = divmod(n, effective)
    sizes = [effective] * full
    if remainder:
        sizes.append(remainder)
    return sizes


# Set once per worker via the pool initializer so ``map_fn`` is pickled once per
# worker rather than once per chunk.
_WORKER_MAP_FN: MapFn[object] | None = None


def _init_worker(map_fn: MapFn[object]) -> None:
    global _WORKER_MAP_FN
    _WORKER_MAP_FN = map_fn


def _apply_worker(task: ChunkTask) -> object:
    assert _WORKER_MAP_FN is not None, "worker map_fn was not initialised"
    n, rng = task
    return _WORKER_MAP_FN(n, rng)


def _iter_processes(
    map_fn: MapFn[T],
    tasks: list[ChunkTask],
    n_workers: int | None,
) -> Iterator[T]:
    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(map_fn,),
    ) as executor:
        # map preserves order, so the reduction matches the sequential run.
        yield from executor.map(_apply_worker, tasks)  # type: ignore[misc]


def _ensure_picklable(obj: object, what: str) -> None:
    try:
        pickle.dumps(obj)
    except Exception as exc:  # noqa: BLE001 - re-raised as a clear error
        raise ValueError(
            f"backend='processes' requires a picklable {what}. Define it at "
            "module level (not as a lambda or a locally-defined closure), and "
            "guard your script entry point with `if __name__ == '__main__':`."
        ) from exc


def map_reduce_chunks(
    n: int,
    *,
    chunk_size: int,
    seed: int,
    map_fn: MapFn[T],
    reduce_fn: ReduceFn[T],
    backend: Backend = "sequential",
    n_workers: int | None = None,
) -> T:
    """Run ``map_fn`` over independently-seeded chunks and fold with ``reduce_fn``.

    ``reduce_fn`` must be associative; the first chunk's result seeds the fold.
    """
    if n_workers is not None and n_workers < 1:
        raise ValueError(f"n_workers must be >= 1, got {n_workers}.")

    sizes = chunk_sizes(n, chunk_size)
    generators = spawn_generators(seed, len(sizes))
    tasks: list[ChunkTask] = list(zip(sizes, generators))

    use_processes = backend == "processes" and len(tasks) > 1
    if backend == "processes":
        _ensure_picklable(map_fn, "map_fn")
        _ensure_picklable(reduce_fn, "reduce_fn")
    if use_processes:
        results: Iterator[T] = _iter_processes(map_fn, tasks, n_workers)
    elif backend in ("sequential", "processes"):
        results = (map_fn(chunk_n, rng) for chunk_n, rng in tasks)
    else:
        raise ValueError(f"Unknown backend {backend!r}.")

    accumulated = next(results)
    for partial in results:
        accumulated = reduce_fn(accumulated, partial)
    return accumulated
