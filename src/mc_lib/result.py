"""The object returned from a simulation run."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

__all__ = ["SimulationResult"]


@dataclass(frozen=True)
class SimulationResult:
    """Aggregated outcome of a Monte Carlo run.

    ``stats`` maps each output column to ``{statistic_label: value}``.  Quantiles
    estimated from a subsample (rather than the full population) are listed in
    ``approximate_columns``.

    The ``converged`` / ``standard_error`` / ``target`` fields are populated only
    by an adaptive :meth:`~mc_lib.engine.Simulation.run_until`; for a fixed-N
    :meth:`~mc_lib.engine.Simulation.run` they stay ``None``.
    """

    n_trials: int
    n_chunks: int
    seed: int
    stats: dict[str, dict[str, float]]
    approximate_columns: frozenset[str]
    converged: bool | None = None
    standard_error: float | None = None
    target: str | None = None

    def __getitem__(self, column: str) -> dict[str, float]:
        return self.stats[column]

    def value(self, column: str, statistic: str) -> float:
        """Return a single statistic, e.g. ``result.value("area", "mean")``."""
        return self.stats[column][statistic]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {col: dict(values) for col, values in self.stats.items()}

    def to_polars(self) -> pl.DataFrame:
        """Tidy long-form table: one row per (column, statistic)."""
        rows = [
            {"column": col, "statistic": stat, "value": value}
            for col, values in self.stats.items()
            for stat, value in values.items()
        ]
        return pl.DataFrame(
            rows,
            schema={"column": pl.String, "statistic": pl.String, "value": pl.Float64},
        )

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        header = (
            f"Monte Carlo simulation: {self.n_trials:,} trials "
            f"in {self.n_chunks} chunk(s), seed={self.seed}"
        )
        if self.converged is not None:
            status = "converged" if self.converged else "stopped at max_trials"
            se = "" if self.standard_error is None else f", se={self.standard_error:.3g}"
            header += f" [adaptive on '{self.target}': {status}{se}]"
        lines = [header]
        for col, values in self.stats.items():
            approx = " (quantiles approximate)" if col in self.approximate_columns else ""
            lines.append(f"  {col}{approx}:")
            for stat, value in values.items():
                lines.append(f"    {stat:<8} = {value:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()
