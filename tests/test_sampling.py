"""Tests for reproducible, lazy sample streams."""

from __future__ import annotations

import numpy as np
import pytest

from polars_mc.distributions import Normal, Uniform
from polars_mc.rng import spawn_generators
from polars_mc.sampling import input_stream, sample_stream


def test_sample_stream_reproducible() -> None:
    s = sample_stream(Normal(0.0, 1.0), seed=5, chunk=100)
    a = np.concatenate(s.take(3).to_list())
    b = np.concatenate(s.take(3).to_list())
    assert np.array_equal(a, b)


def test_sample_stream_different_seed_differs() -> None:
    a = np.concatenate(sample_stream(Normal(0.0, 1.0), seed=1, chunk=100).take(2).to_list())
    b = np.concatenate(sample_stream(Normal(0.0, 1.0), seed=2, chunk=100).take(2).to_list())
    assert not np.array_equal(a, b)


def test_sample_stream_chunk_shape() -> None:
    chunks = sample_stream(Uniform(0.0, 1.0), seed=0, chunk=64).take(3).to_list()
    assert all(c.shape == (64,) for c in chunks)


def test_sample_stream_matches_spawn_generators() -> None:
    # The k-th streamed chunk must equal the k-th batch generator's draw: the
    # streaming layer shares the batch engine's SeedSequence reproducibility.
    dist = Normal(2.0, 3.0)
    batch = [dist.sample(g, 100) for g in spawn_generators(9, 4)]
    streamed = sample_stream(dist, seed=9, chunk=100).take(4).to_list()
    for expected, got in zip(batch, streamed):
        assert np.array_equal(expected, got)


def test_input_stream_reproducible_and_shaped() -> None:
    s = input_stream({"x": Normal(0.0, 1.0), "y": Uniform(0.0, 1.0)}, seed=7, chunk=50)
    first = s.take(2).to_list()
    second = s.take(2).to_list()
    assert np.array_equal(first[0]["x"], second[0]["x"])
    assert np.array_equal(first[1]["y"], second[1]["y"])
    assert first[0]["x"].shape == (50,)


def test_invalid_chunk_raises() -> None:
    with pytest.raises(ValueError):
        sample_stream(Normal(0.0, 1.0), chunk=0)
    with pytest.raises(ValueError):
        input_stream({"x": Normal(0.0, 1.0)}, chunk=0)
    with pytest.raises(ValueError):
        input_stream({}, chunk=10)
