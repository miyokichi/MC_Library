"""Result objects for statistics and simulations."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

__all__ = ["StatResult", "SimulationResult"]


@dataclass(frozen=True)
class StatResult:
    """Aggregated statistics over a set of observations.

    ``stats`` maps each column to ``{statistic_label: value}``.  Quantiles
    estimated from a bounded reservoir (rather than the full data) are listed in
    ``approximate_columns``.
    """

    n: int
    stats: dict[str, dict[str, float]]
    approximate_columns: frozenset[str]

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

    def _summary_header(self) -> str:
        return f"Statistics over {self.n:,} observations"

    def summary(self) -> str:
        """Human-readable multi-line summary."""
        lines = [self._summary_header()]
        for col, values in self.stats.items():
            approx = " (quantiles approximate)" if col in self.approximate_columns else ""
            lines.append(f"  {col}{approx}:")
            for stat, value in values.items():
                lines.append(f"    {stat:<8} = {value:.6g}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


@dataclass(frozen=True)
class SimulationResult(StatResult):
    """A :class:`StatResult` from a Monte Carlo run, with run metadata."""

    n_chunks: int = 0
    seed: int = 0

    @property
    def n_trials(self) -> int:
        return self.n

    def _summary_header(self) -> str:
        return (
            f"Monte Carlo simulation: {self.n:,} trials "
            f"in {self.n_chunks} chunk(s), seed={self.seed}"
        )
