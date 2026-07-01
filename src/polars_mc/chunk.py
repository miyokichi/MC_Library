"""Apply a user trial to one chunk of sampled inputs.

This wires the sampling layer to the reducer: draw the chunk's inputs, run the
trial (array or frame style), and reduce the result to per-column accumulators.
The heavy rows never leave the chunk -- only the small accumulators do.

Two trial styles are supported:

* **array style** -- an ordinary function whose parameters are input names; it
  receives NumPy arrays (length ``n``) and returns ``{output_name: array}``.
* **frame style** -- ``(pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame`` built
  from Polars expressions.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .aggregate import ColumnAccumulator, ColumnPlan
from .distributions import Distribution
from .reduce import accumulate_columns
from .sampling import sample_arrays

__all__ = ["TrialFn", "TrialStyle", "classify_trial", "run_chunk"]

# Either an array-style function ``(**input_arrays) -> {name: array}`` or a
# frame-style function ``(LazyFrame) -> LazyFrame | DataFrame``.
TrialFn = Callable[..., Any]
TrialStyle = Literal["array", "frame"]


def classify_trial(trial: TrialFn, input_names: set[str]) -> TrialStyle:
    """Infer the trial style from its signature.

    A trial whose parameters are all input names (or which accepts ``**kwargs``)
    is treated as array style; anything else is frame style.
    """
    try:
        sig = inspect.signature(trial)
    except (TypeError, ValueError):
        return "frame"

    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    named = {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    }
    if accepts_var_kw or (named and named <= input_names):
        return "array"
    return "frame"


def _coerce_output(name: str, value: object, n: int) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(n, array)
    if array.shape[0] != n:
        raise ValueError(
            f"trial output {name!r} has length {array.shape[0]}, expected {n}."
        )
    return array


def _apply_array_trial(
    trial: TrialFn,
    arrays: dict[str, NDArray[np.generic]],
    n: int,
) -> pl.DataFrame:
    sig = inspect.signature(trial)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    kwargs = (
        arrays
        if accepts_var_kw
        else {name: arrays[name] for name in sig.parameters if name in arrays}
    )
    result = trial(**kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            "An array-style trial must return a dict of {output_name: array}, "
            f"got {type(result).__name__}."
        )
    columns: dict[str, NDArray[np.generic]] = dict(arrays)
    for name, value in result.items():
        columns[name] = _coerce_output(name, value, n)
    return pl.DataFrame(columns)


def _apply_frame_trial(
    trial: TrialFn,
    arrays: dict[str, NDArray[np.generic]],
) -> pl.DataFrame:
    result = trial(pl.DataFrame(arrays).lazy())
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if isinstance(result, pl.DataFrame):
        return result
    raise TypeError(
        "A frame-style trial must return a polars LazyFrame or DataFrame, got "
        f"{type(result).__name__}."
    )


def run_chunk(
    inputs: dict[str, Distribution],
    trial: TrialFn,
    plans: tuple[ColumnPlan, ...],
    n: int,
    rng: np.random.Generator,
    capacity: int,
    style: TrialStyle,
) -> dict[str, ColumnAccumulator]:
    """Run one chunk and return per-column accumulators."""
    arrays = sample_arrays(inputs, n, rng)
    if style == "array":
        out = _apply_array_trial(trial, arrays, n)
    else:
        out = _apply_frame_trial(trial, arrays)
    return accumulate_columns(out, plans, rng, capacity)
