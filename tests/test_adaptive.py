"""Tests for constraint_theory.adaptive — adaptive_epsilon, classify_region, AdaptiveTolerance, manifold helpers."""

import math
import pytest
from constraint_theory.adaptive import (
    adaptive_epsilon, classify_region, ManifoldRegion,
    AdaptiveTolerance, curvature_from_manifold, manifold_distance,
    adaptive_snap, DEFAULT_K, DEFAULT_EPSILON_MAX, DEFAULT_EPSILON_MIN,
)


class TestAdaptiveEpsilon:
    def test_basic(self):
        eps = adaptive_epsilon(curvature=10.0)
        assert DEFAULT_EPSILON_MIN <= eps <= DEFAULT_EPSILON_MAX

    def test_high_curvature_tight(self):
        eps_high = adaptive_epsilon(curvature=100.0)
        eps_low = adaptive_epsilon(curvature=1.0)
        assert eps_high < eps_low

    def test_zero_curvature(self):
        # curvature=0 should return epsilon_max (since raw = k/0 → inf, clamped)
        eps = adaptive_epsilon(curvature=0.0)
        assert eps == DEFAULT_EPSILON_MAX

    def test_negative_curvature(self):
        with pytest.raises(ValueError):
            adaptive_epsilon(curvature=-1.0)

    def test_nan_curvature(self):
        with pytest.raises(ValueError):
            adaptive_epsilon(curvature=float("nan"))

    def test_custom_k(self):
        eps = adaptive_epsilon(curvature=10.0, k=2.0)
        assert abs(eps - 0.2) < 0.01

    def test_clamp_to_max(self):
        eps = adaptive_epsilon(curvature=0.1, epsilon_max=0.01)
        assert eps == 0.01

    def test_clamp_to_min(self):
        eps = adaptive_epsilon(curvature=1e15)
        assert eps == DEFAULT_EPSILON_MIN


class TestClassifyRegion:
    def test_far(self):
        assert classify_region(0.001) == ManifoldRegion.FAR

    def test_approaching(self):
        assert classify_region(0.5) == ManifoldRegion.APPROACHING

    def test_near(self):
        assert classify_region(5.0) == ManifoldRegion.NEAR

    def test_critical(self):
        assert classify_region(50.0) == ManifoldRegion.CRITICAL

    def test_singular(self):
        assert classify_region(float("inf")) == ManifoldRegion.SINGULAR


class TestAdaptiveTolerance:
    def test_call(self):
        at = AdaptiveTolerance(k=1.0)
        eps = at(10.0)
        assert DEFAULT_EPSILON_MIN <= eps <= DEFAULT_EPSILON_MAX

    def test_batch(self):
        at = AdaptiveTolerance(k=1.0)
        results = at.batch([1.0, 10.0, 100.0])
        assert len(results) == 3
        # Higher curvature → lower epsilon
        assert results[2] < results[1] < results[0]

    def test_caching(self):
        at = AdaptiveTolerance()
        at(5.0)
        at(5.0)  # should hit cache
        stats = at.cache_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_clear_cache(self):
        at = AdaptiveTolerance()
        at(5.0)
        at.clear_cache()
        stats = at.cache_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0


class TestManifoldDistance:
    def test_distance_to_origin(self):
        d = manifold_distance(1.0, 0.0, [(0.0, 0.0)])
        assert abs(d - 1.0) < 1e-6

    def test_distance_empty_boundary(self):
        d = manifold_distance(0.0, 0.0, [])
        assert d == float("inf")

    def test_nearest_boundary(self):
        d = manifold_distance(0.0, 0.0, [(10.0, 0.0), (1.0, 0.0)])
        assert abs(d - 1.0) < 1e-6


class TestCurvatureFromManifold:
    def test_flat_metric(self):
        def flat_metric(x, y):
            return x * x + y * y
        c = curvature_from_manifold(0.0, 0.0, flat_metric)
        assert c >= 0.0

    def test_returns_positive(self):
        def metric(x, y):
            return math.sin(x) * math.cos(y)
        c = curvature_from_manifold(0.5, 0.5, metric)
        assert c >= 0.0


class TestAdaptiveSnap:
    def test_returns_tuple(self):
        eps, d, ok = adaptive_snap(0.5, 0.5, [(0.0, 0.0)])
        assert isinstance(eps, float)
        assert isinstance(d, float)
        assert isinstance(ok, bool)
        assert eps > 0.0
        assert d > 0.0
