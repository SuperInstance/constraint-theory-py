"""
Adaptive Tolerance — epsilon(c) = k / c for manifold boundary regions.

In constraint theory, the manifold boundary is where the local curvature
approaches infinity and the linearised constraint becomes unreliable.
Adaptive tolerance adjusts the snapping precision to account for this:

    ε(c) = min(k / c, ε_max)

where *c* is the local curvature and *k* is a configurable constant.

This module provides the core formula, region classification, and
a convenience compositor that wraps an existing epsilon provider.

Example
-------
>>> from constraint_theory.adaptive import adaptive_epsilon, classify_region
>>> eps = adaptive_epsilon(curvature=10.0, k=0.5)
>>> 0.0 < eps < 0.1
True
>>> classify_region(curvature=0.01)
'far'
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, List, Optional, Sequence, Tuple, Union


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------

DEFAULT_K: float = 1.0
"""Default proportionality constant."""

DEFAULT_EPSILON_MAX: float = 0.5
"""Default maximum allowed epsilon (prevents divergence near zero curvature)."""

DEFAULT_EPSILON_MIN: float = 1e-12
"""Default minimum epsilon clamp to avoid division by zero."""

# Curvature thresholds (inverse distance to boundary)
_FAR_THRESHOLD: float = 0.01
_NEAR_THRESHOLD: float = 1.0
_CRITICAL_THRESHOLD: float = 10.0


def adaptive_epsilon(
    curvature: float,
    k: float = DEFAULT_K,
    epsilon_max: float = DEFAULT_EPSILON_MAX,
    epsilon_min: float = DEFAULT_EPSILON_MIN,
) -> float:
    """Compute epsilon = min(k / c, ε_max), clamped to [ε_min, ε_max].

    Parameters
    ----------
    curvature: float
        Local curvature of the manifold at the evaluation point (c ≥ 0).
    k: float
        Proportionality constant (default 1.0).
    epsilon_max: float
        Maximum allowed epsilon (default 0.5).
    epsilon_min: float
        Minimum epsilon clamp (default 1e-12).

    Returns
    -------
    float
        The adaptive tolerance value ε(c) ∈ [ε_min, ε_max].

    Raises
    ------
    ValueError
        If curvature < 0 or any argument is NaN.

    Example
    -------
    >>> eps = adaptive_epsilon(100.0)
    >>> eps  # 1.0 / 100.0
    0.01
    """
    if math.isnan(curvature) or math.isnan(k) or math.isnan(epsilon_max) or math.isnan(epsilon_min):
        raise ValueError("NaN argument")
    if curvature < 0.0:
        raise ValueError(f"curvature must be >= 0, got {curvature}")

    if curvature < epsilon_min / k:
        return epsilon_max

    raw: float = k / curvature
    return max(epsilon_min, min(raw, epsilon_max))


# ---------------------------------------------------------------------------
# Region classification
# ---------------------------------------------------------------------------


class ManifoldRegion(str, Enum):
    """Classification of a point relative to the manifold boundary."""
    FAR = "far"
    APPROACHING = "approaching"
    NEAR = "near"
    CRITICAL = "critical"
    SINGULAR = "singular"


def classify_region(
    curvature: float,
    far_threshold: float = _FAR_THRESHOLD,
    near_threshold: float = _NEAR_THRESHOLD,
    critical_threshold: float = _CRITICAL_THRESHOLD,
) -> ManifoldRegion:
    """Classify a manifold region based on local curvature.

    Threshold hierarchy: ``far < approaching < near < critical < singular``.

    Parameters
    ----------
    curvature: float
        Local curvature (≥ 0).
    far_threshold: float
        Max curvature for 'far' region (default 0.01).
    near_threshold: float
        Max curvature for 'approaching' region (default 1.0).
    critical_threshold: float
        Max curvature for 'near' region (default 10.0).

    Returns
    -------
    ManifoldRegion
        One of FAR, APPROACHING, NEAR, CRITICAL, or SINGULAR.

    Example
    -------
    >>> classify_region(0.001)
    <ManifoldRegion.FAR: 'far'>
    >>> classify_region(100.0)
    <ManifoldRegion.SINGULAR: 'singular'>
    """
    if curvature <= far_threshold:
        return ManifoldRegion.FAR
    if curvature <= near_threshold:
        return ManifoldRegion.APPROACHING
    if curvature <= critical_threshold:
        return ManifoldRegion.NEAR
    if curvature < float("inf"):
        return ManifoldRegion.CRITICAL
    return ManifoldRegion.SINGULAR


# ---------------------------------------------------------------------------
# Adaptive tolerance compositor
# ---------------------------------------------------------------------------


class AdaptiveTolerance:
    """Composable adaptive tolerance for use in constraint solvers.

    Wraps the core ``adaptive_epsilon`` function with caching and
    batch computation.

    Parameters
    ----------
    k: float
        Proportionality constant (default 1.0).
    epsilon_max: float
        Maximum epsilon (default 0.5).
    epsilon_min: float
        Minimum epsilon (default 1e-12).
    fallback: callable or None
        Fallback epsilon function for unclassified points (optional).

    Example
    -------
    >>> at = AdaptiveTolerance(k=0.5)
    >>> at(100.0)
    0.005
    >>> at.batch([1.0, 10.0, 100.0])
    [0.5, 0.05, 0.005]
    """

    def __init__(
        self,
        k: float = DEFAULT_K,
        epsilon_max: float = DEFAULT_EPSILON_MAX,
        epsilon_min: float = DEFAULT_EPSILON_MIN,
        fallback: Optional[Callable[[float], float]] = None,
    ) -> None:
        self.k = k
        self.epsilon_max = epsilon_max
        self.epsilon_min = epsilon_min
        self.fallback = fallback

        self._cache: dict[float, float] = {}
        self._hit_count: int = 0
        self._miss_count: int = 0
        self._max_cache: int = 1024

    def __call__(self, curvature: float) -> float:
        """Compute adaptive epsilon for a given curvature.

        Parameters
        ----------
        curvature: float
            Local curvature.

        Returns
        -------
        float
            Adaptive epsilon value.
        """
        if curvature in self._cache:
            self._hit_count += 1
            return self._cache[curvature]

        self._miss_count += 1
        result: float = adaptive_epsilon(
            curvature, k=self.k, epsilon_max=self.epsilon_max, epsilon_min=self.epsilon_min
        )

        if len(self._cache) < self._max_cache:
            self._cache[curvature] = result

        return result

    def batch(self, curvatures: Sequence[float]) -> List[float]:
        """Compute epsilon for multiple curvatures at once.

        Parameters
        ----------
        curvatures: sequence of float
            Curvatures to evaluate.

        Returns
        -------
        list of float
            Adaptive epsilon values.
        """
        return [self(c) for c in curvatures]

    def clear_cache(self) -> None:
        """Reset the internal cache."""
        self._cache.clear()
        self._hit_count = 0
        self._miss_count = 0

    def cache_stats(self) -> dict[str, int]:
        """Return current cache hit/miss statistics."""
        return {
            "size": len(self._cache),
            "max_size": self._max_cache,
            "hits": self._hit_count,
            "misses": self._miss_count,
            "hit_rate": (
                self._hit_count / (self._hit_count + self._miss_count)
                if (self._hit_count + self._miss_count) > 0
                else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Curvature estimation utilities
# ---------------------------------------------------------------------------


def curvature_from_manifold(
    x: float, y: float,
    metric: Callable[[float, float], float],
    delta: float = 1e-6,
) -> float:
    """Estimate local curvature from a metric function via finite differences.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.
    metric: callable (x, y) -> float
        Metric function to evaluate curvature on.
    delta: float
        Finite-difference step (default 1e-6).

    Returns
    -------
    float
        Estimated curvature.
    """
    g_xx: float = (metric(x + delta, y) - 2 * metric(x, y) + metric(x - delta, y)) / (delta * delta)
    g_yy: float = (metric(x, y + delta) - 2 * metric(x, y) + metric(x, y - delta)) / (delta * delta)
    g_xy: float = (
        metric(x + delta, y + delta)
        - metric(x + delta, y - delta)
        - metric(x - delta, y + delta)
        + metric(x - delta, y - delta)
    ) / (4.0 * delta * delta)
    return abs(g_xx * g_yy - g_xy * g_xy)


# ---------------------------------------------------------------------------
# Module __all__
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Manifold distance & adaptive snap convenience functions
# ---------------------------------------------------------------------------

# Type alias: a point in 2-D
_ManifoldPoint = Tuple[float, float]


def manifold_distance(
    x: float,
    y: float,
    boundary_points: Sequence[Tuple[float, float]],
) -> float:
    """Compute the Euclidean distance from a point to the nearest boundary point.

    In constraint theory, this distance is used as a proxy for local
    curvature: points close to the boundary have high curvature, and the
    adaptive tolerance should be tighter (smaller epsilon).

    Parameters
    ----------
    x: float
        X coordinate of the query point.
    y: float
        Y coordinate of the query point.
    boundary_points: sequence of (float, float)
        Set of points defining the manifold boundary.

    Returns
    -------
    float
        Minimum Euclidean distance to any boundary point.

    Example
    -------
    >>> d = manifold_distance(0.0, 0.0, [(1.0, 0.0), (0.0, 1.0)])
    >>> d  # distance to nearest boundary point
    1.0
    """
    best: float = float("inf")
    for bx, by in boundary_points:
        dx: float = x - bx
        dy: float = y - by
        dist: float = dx * dx + dy * dy
        if dist < best:
            best = dist
    return math.sqrt(best) if best != float("inf") else float("inf")


def adaptive_snap(
    x: float,
    y: float,
    boundary_points: Sequence[Tuple[float, float]],
    k: float = DEFAULT_K,
) -> Tuple[float, float, bool]:
    """Snap a point with an adaptive tolerance based on manifold distance.

    Distance to the nearest boundary point is computed, then converted
    to a curvature proxy (1 / distance), which drives epsilon.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.
    boundary_points: sequence of (float, float)
        Boundary-defining points for manifold distance.
    k: float
        Proportionality constant for adaptive epsilon (default 1.0).

    Returns
    -------
    tuple of (adapted_epsilon, boundary_distance, is_within_tolerance)
        - adapted_epsilon: the tolerance value used
        - boundary_distance: distance to nearest boundary point
        - is_within_tolerance: True if the point would be accepted
          under the adapted epsilon (snap error ≤ epsilon)

    Note
    ----
    This function does not actually snap to the lattice — use
    ``eisenstein.snap`` separately.  It computes the *tolerance*
    that would be used by a downstream snapping step.

    Example
    -------
    >>> eps, d, ok = adaptive_snap(0.5, 0.5, [(0.0, 0.0)])
    >>> eps > 0.0 and d > 0.0
    True
    >>> isinstance(ok, bool)
    True
    """
    dist: float = manifold_distance(x, y, boundary_points)
    # Avoid division by zero — clamp curvature to a large but finite value
    curvature: float = 1.0 / max(dist, 1e-300)
    eps: float = adaptive_epsilon(curvature, k=k)
    # Use the eisenstein module to snap and compare
    # (lazy import to avoid circular dependency at module level)
    from constraint_theory.eisenstein import snap_with_error
    _, snap_err = snap_with_error(x, y)
    return eps, dist, snap_err <= eps


__all__ = [
    "ManifoldRegion",
    "AdaptiveTolerance",
    "adaptive_epsilon",
    "classify_region",
    "curvature_from_manifold",
    "manifold_distance",
    "adaptive_snap",
    "DEFAULT_K",
    "DEFAULT_EPSILON_MAX",
    "DEFAULT_EPSILON_MIN",
]
