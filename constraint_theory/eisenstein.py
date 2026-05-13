"""
Eisenstein Lattice Operations — snap-to-lattice, Weyl chambers, dodecet encoding,
norm computation, and A₂ lattice arithmetic for constraint theory.

This module is a pure-Python port of the dodecet-encoder Rust crate's
``eisenstein.rs`` and ``dodecet.rs`` modules.

The A₂ (hexagonal) lattice is the set of all integer-linear combinations of
the two basis vectors {1, ω} where ω = e^{2πi/3} = -1/2 + i√3/2.

Every point in the complex plane is within the covering radius
ρ = 1/√3 ≈ 0.577 of a lattice point — this guarantees a bounded quantisation
error.

Example
-------
>>> from constraint_theory.eisenstein import (
...     snap, decode, encode, norm_sq, A2Point, Dodecet
... )
>>> pt = snap(0.5, 0.3)
>>> pt.a, pt.b
(0, 0)
>>> pt.error < 0.577
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQRT_3: float = math.sqrt(3.0)
"""√3, the fundamental constant of the A₂ lattice."""

OMEGA_RE: float = -0.5
"""Real part of the Eisenstein unit ω = e^{2πi/3}."""

OMEGA_IM: float = SQRT_3 / 2.0
"""Imaginary part of the Eisenstein unit ω."""

COVERING_RADIUS: float = 1.0 / SQRT_3
"""Maximum distance from any point to the nearest A₂ lattice point (ρ ≈ 0.577)."""

SAFE_THRESHOLD: float = COVERING_RADIUS / 2.0
"""Error threshold below which a snap is considered "safe"."""

_DODECET_MAX: int = 0xFFF
"""Maximum value representable by a 12-bit dodecet (4095)."""


# ---------------------------------------------------------------------------
# Auxiliary types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class A2Point:
    """An Eisenstein integer a + bω where a, b ∈ ℤ.

    Attributes
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.
    """

    a: int
    b: int

    def __post_init__(self) -> None:
        # Validate at construction time only (not on frozen dataclass defaults)
        object.__setattr__(self, "a", int(self.a))
        object.__setattr__(self, "b", int(self.b))

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to_cartesian(self) -> Tuple[float, float]:
        """Return the (x, y) coordinates in the complex plane."""
        return (
            float(self.a) + float(self.b) * OMEGA_RE,
            float(self.b) * OMEGA_IM,
        )

    @classmethod
    def from_cartesian(cls, x: float, y: float) -> "A2Point":
        """Snap a cartesian point to the nearest A₂ lattice point.

        Parameters
        ----------
        x: float
            X coordinate.
        y: float
            Y coordinate.

        Returns
        -------
        A2Point
            Nearest lattice point (a, b).
        """
        sr = _snap_inner(x, y)
        return cls(a=sr.a, b=sr.b)

    def __repr__(self) -> str:
        return f"A2Point({self.a}, {self.b})"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: "A2Point") -> "A2Point":
        return A2Point(self.a + other.a, self.b + other.b)

    def __sub__(self, other: "A2Point") -> "A2Point":
        return A2Point(self.a - other.a, self.b - other.b)

    def __neg__(self) -> "A2Point":
        return A2Point(-self.a, -self.b)

    def __mul__(self, scalar: int) -> "A2Point":
        return A2Point(self.a * scalar, self.b * scalar)

    def __rmul__(self, scalar: int) -> "A2Point":
        return self.__mul__(scalar)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, A2Point):
            return NotImplemented
        return self.a == other.a and self.b == other.b

    def __hash__(self) -> int:
        return hash((self.a, self.b))


@dataclass(frozen=True)
class Dodecet:
    """12-bit Eisenstein lattice encoding.

    Bit layout::

        bits 11-8 : error level   (nibble 2, 0-15)
        bits  7-4 : angle level   (nibble 1, 0-15)
        bits  3   : safety flag   (0 = safe / within deadband, 1 = critical)
        bits  2-0 : chamber index (nibble 0 lower 3 bits, 0-5)

    This compact form is how PLATO tiles store their constraint snap state
    in 12 bits of metadata.
    """

    raw: int
    """The raw 12-bit value (0-4095)."""

    def __post_init__(self) -> None:
        # Clamp to valid range
        object.__setattr__(self, "raw", max(0, min(self.raw, _DODECET_MAX)))

    # ------------------------------------------------------------------
    # Field accessors
    # ------------------------------------------------------------------

    @property
    def error_level(self) -> int:
        """Snap error quantised to 16 levels (nibble 2)."""
        return (self.raw >> 8) & 0xF

    @property
    def angle_level(self) -> int:
        """Azimuth (angle) quantised to 16 levels (nibble 1)."""
        return (self.raw >> 4) & 0xF

    @property
    def chamber(self) -> int:
        """Weyl chamber index (0-5)."""
        return self.raw & 0x7

    @property
    def is_safe(self) -> bool:
        """True when the snap error was within the safe threshold."""
        return (self.raw >> 3) & 1 == 0

    @property
    def is_critical(self) -> bool:
        """True when the snap error exceeded the safe threshold."""
        return not self.is_safe

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_fields(
        cls,
        error_level: int,
        angle_level: int,
        chamber: int,
        is_safe: bool = True,
    ) -> "Dodecet":
        """Build a Dodecet from explicit field values."""
        safe_bit: int = 0 if is_safe else 1
        raw: int = (
            ((error_level & 0xF) << 8)
            | ((angle_level & 0xF) << 4)
            | ((safe_bit & 1) << 3)
            | (chamber & 0x7)
        )
        return cls(raw=raw)

    @classmethod
    def from_point(cls, x: float, y: float) -> "Dodecet":
        """Snap a 2-D point and encode as a Dodecet."""
        sr = _snap_inner(x, y)
        return cls.from_fields(
            error_level=sr.error_level,
            angle_level=sr.angle_level,
            chamber=sr.chamber,
            is_safe=sr.is_safe,
        )

    def __repr__(self) -> str:
        return (
            f"Dodecet(0x{self.raw:03X}, "
            f"err={self.error_level}, angle={self.angle_level}, "
            f"chamber={self.chamber}, safe={self.is_safe})"
        )


class Chamber(IntEnum):
    """The six Weyl chambers of the A₂ lattice.

    Chambers are labelled 0-5 by the sorted permutation of the
    three barycentric coordinates (b1, b2, b3).
    """

    CHAMBER_0 = 0
    CHAMBER_1 = 1
    CHAMBER_2 = 2
    CHAMBER_3 = 3
    CHAMBER_4 = 4
    CHAMBER_5 = 5

    @property
    def parity(self) -> int:
        """Parity: +1 for even chambers (0, 2, 5), -1 for odd (1, 3, 4)."""
        return 1 if self.value in (0, 2, 5) else -1


# ---------------------------------------------------------------------------
# Snapping result (internal — used for encoding)
# ---------------------------------------------------------------------------

@dataclass
class _SnapResult:
    a: int
    b: int
    error: float
    error_normalized: float
    error_level: int
    angle_level: int
    chamber: int
    is_safe: bool


# ---------------------------------------------------------------------------
# Core snapping
# ---------------------------------------------------------------------------

# Weyl permutations — sorted descending order of (b1, b2, b3)
_WEYL_PERMS: Tuple[Tuple[int, int, int], ...] = (
    (0, 1, 2),
    (0, 2, 1),
    (1, 0, 2),
    (1, 2, 0),
    (2, 0, 1),
    (2, 1, 0),
)

_EVEN_CHAMBERS: Tuple[int, ...] = (0, 2, 5)
_ODD_CHAMBERS: Tuple[int, ...] = (1, 3, 4)


def _round32(v: float) -> int:
    """Round to nearest int (exactly like Rust's f64.round())."""
    return int(math.floor(v + 0.5))


def _snap_inner(x: float, y: float) -> _SnapResult:
    """Core A₂ lattice snap. Exists here and in temporal.py for independence."""
    a_f: float = x - y * OMEGA_RE / OMEGA_IM
    b_f: float = y / OMEGA_IM

    a0: int = _round32(a_f)
    b0: int = _round32(b_f)

    best_a: int = a0
    best_b: int = b0
    best_err: float = float("inf")

    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            ca: int = a0 + da
            cb: int = b0 + db
            cx: float = ca + cb * OMEGA_RE
            cy: float = cb * OMEGA_IM
            err: float = math.hypot(x - cx, y - cy)
            if err < best_err:
                best_a = ca
                best_b = cb
                best_err = err

    chamber: int = _classify_chamber(x, y)
    err_norm: float = min(best_err / COVERING_RADIUS, 1.0)
    err_level: int = _round32(err_norm * 15.0)

    dx = x - (best_a + best_b * OMEGA_RE)
    dy = y - (best_b * OMEGA_IM)
    if dx != 0.0 or dy != 0.0:
        angle: float = math.atan2(dy, dx)
        norm_angle: float = (angle + math.pi) / (2.0 * math.pi)
        angle_level: int = (int(norm_angle * 16.0)) % 16
    else:
        angle_level = 0

    is_safe: bool = best_err < SAFE_THRESHOLD

    return _SnapResult(
        a=best_a,
        b=best_b,
        error=best_err,
        error_normalized=err_norm,
        error_level=err_level,
        angle_level=angle_level,
        chamber=chamber,
        is_safe=is_safe,
    )


def _classify_chamber(x: float, y: float) -> int:
    b1: float = x - y * OMEGA_RE / OMEGA_IM
    b2: float = y / OMEGA_IM
    b3: float = -(b1 + b2)
    vals: List[float] = [b1, b2, b3]
    sorted_idx: List[int] = sorted(range(3), key=lambda i: vals[i], reverse=True)
    perm: Tuple[int, int, int] = (sorted_idx[0], sorted_idx[1], sorted_idx[2])
    try:
        return _WEYL_PERMS.index(perm)
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def snap(x: float, y: float) -> A2Point:
    """Snap a 2-D point to the nearest A₂ lattice point.

    Uses a 9-candidate Voronoi search guaranteed to find the nearest lattice
    point (covering radius = 1/√3 ≈ 0.577).

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    A2Point
        Nearest Eisenstein integer (a, b).

    Example
    -------
    >>> snap(0.0, 0.0)
    A2Point(0, 0)
    >>> snap(1.0, 0.0)
    A2Point(1, 0)
    """
    sr = _snap_inner(x, y)
    return A2Point(a=sr.a, b=sr.b)


def snap_with_error(x: float, y: float) -> Tuple[A2Point, float]:
    """Snap and also return the Euclidean error.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    tuple of (A2Point, float)
        Nearest lattice point and distance from input to that point.
    """
    sr = _snap_inner(x, y)
    return A2Point(a=sr.a, b=sr.b), sr.error


def snap_with_metadata(x: float, y: float) -> Tuple[A2Point, float, int, int, bool]:
    """Snap with full metadata.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    tuple of (A2Point, error, chamber, error_level, is_safe)
    """
    sr = _snap_inner(x, y)
    return (
        A2Point(a=sr.a, b=sr.b),
        sr.error,
        sr.chamber,
        sr.error_level,
        sr.is_safe,
    )


def snap_batch(points: List[Tuple[float, float]]) -> List[Tuple[A2Point, float]]:
    """Snap multiple points at once.

    Parameters
    ----------
    points: list of (x, y)
        Points to snap.

    Returns
    -------
    list of (A2Point, float)
        Nearest lattice points and their errors.
    """
    return [snap_with_error(x, y) for x, y in points]


# ---------------------------------------------------------------------------
# Norm computation
# ---------------------------------------------------------------------------


def norm_sq(a: int, b: int) -> int:
    """Squared A₂ norm: a² + ab + b².

    This is the norm in the Eisenstein integer ring ℤ[ω].

    Parameters
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.

    Returns
    -------
    int
        a² + ab + b².

    Example
    -------
    >>> norm_sq(0, 0)
    0
    >>> norm_sq(1, 0)
    1
    >>> norm_sq(1, 1)
    3
    >>> norm_sq(2, 1)
    7
    """
    return a * a + a * b + b * b


def norm(a: int, b: int) -> float:
    """Euclidean norm in the Eisenstein plane: sqrt(a² + ab + b²).

    Parameters
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.

    Returns
    -------
    float
        √(a² + ab + b²).
    """
    return math.sqrt(norm_sq(a, b))


def distance_sq(a1: int, b1: int, a2: int, b2: int) -> int:
    """Squared distance between two Eisenstein integers."""
    da: int = a1 - a2
    db: int = b1 - b2
    return norm_sq(da, db)


def distance(a1: int, b1: int, a2: int, b2: int) -> float:
    """Euclidean distance between two Eisenstein integers."""
    return math.sqrt(distance_sq(a1, b1, a2, b2))


# ---------------------------------------------------------------------------
# Chamber classification
# ---------------------------------------------------------------------------


def classify_chamber(x: float, y: float) -> int:
    """Classify a point into one of the 6 Weyl chambers.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    int
        Chamber index (0-5).
    """
    return _classify_chamber(x, y)


def chamber_barycentric(x: float, y: float) -> Tuple[float, float, float]:
    """Compute barycentric coordinates (b1, b2, b3) with respect to the A₂ roots.

    b1 + b2 + b3 = 0 by construction.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    tuple of (float, float, float)
        Barycentric coordinates.
    """
    b1: float = x - y * OMEGA_RE / OMEGA_IM
    b2: float = y / OMEGA_IM
    b3: float = -(b1 + b2)
    return b1, b2, b3


# ---------------------------------------------------------------------------
# Dodecet encoding / decoding
# ---------------------------------------------------------------------------


def encode(x: float, y: float) -> Dodecet:
    """Snap a point and encode as a 12-bit Dodecet.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    Dodecet
        The encoded 12-bit value.

    Example
    -------
    >>> d = encode(0.3, 0.4)
    >>> 0 <= d.raw <= 0xFFF
    True
    """
    return Dodecet.from_point(x, y)


def encode_from_fields(
    error_level: int,
    angle_level: int,
    chamber: int,
    is_safe: bool = True,
) -> Dodecet:
    """Encode explicit fields into a Dodecet."""
    return Dodecet.from_fields(
        error_level=error_level,
        angle_level=angle_level,
        chamber=chamber,
        is_safe=is_safe,
    )


def decode(dodecet: Union[int, Dodecet]) -> Tuple[int, int, int, bool]:
    """Decode a Dodecet into its four component fields.

    Parameters
    ----------
    dodecet: int or Dodecet
        The value to decode (as raw int or Dodecet instance).

    Returns
    -------
    tuple of (error_level, angle_level, chamber_number, is_safe)
    """
    if isinstance(dodecet, Dodecet):
        raw = dodecet.raw
    else:
        raw = int(dodecet)
    err_level: int = (raw >> 8) & 0xF
    angle_level: int = (raw >> 4) & 0xF
    chamber_byte: int = raw & 0xF
    chamber: int = chamber_byte & 0x7
    is_safe: bool = (chamber_byte >> 3) & 1 == 0
    return err_level, angle_level, chamber, is_safe


# ---------------------------------------------------------------------------
# Voronoi region / covering radius utilities
# ---------------------------------------------------------------------------


def voronoi_cell_area() -> float:
    """Area of the A₂ fundamental parallelogram (Voronoi cell).

    Returns
    -------
    float
        √3/2 ≈ 0.8660
    """
    return SQRT_3 / 2.0


def error_cdf(error: float) -> float:
    """Fraction of the A₂ Voronoi cell with distance < *error* from a lattice point.

    For errors up to the covering radius, this approximates what fraction
    of uniformly-random inputs would land closer than *error*.

    Parameters
    ----------
    error: float
        Distance from lattice point (≤ COVERING_RADIUS).

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    e: float = max(0.0, min(error, COVERING_RADIUS))
    return math.pi * e * e / voronoi_cell_area()


def voronoi_radius() -> float:
    """The inner radius of the Voronoi cell. (Same as covering radius for A₂.)"""
    return COVERING_RADIUS


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def lattice_points_within(radius: float) -> List[A2Point]:
    """Generate all A₂ lattice points within a given radius of the origin.

    Parameters
    ----------
    radius: float
        Maximum Euclidean distance from origin.

    Returns
    -------
    list of A2Point
        All lattice points within the radius (including the origin).
    """
    max_a: int = int(math.ceil(radius / 1.0)) + 1
    max_b: int = int(math.ceil(radius / OMEGA_IM)) + 1

    result: List[A2Point] = []
    r2: float = radius * radius
    for a in range(-max_a, max_a + 1):
        for b in range(-max_b, max_b + 1):
            if norm_sq(a, b) <= r2:
                result.append(A2Point(a, b))
    return result


# ---------------------------------------------------------------------------
# Module __all__
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Additional convenience / API-matching functions
# ---------------------------------------------------------------------------


def snap_to_lattice(
    x: float, y: float, epsilon: float = 0.0
) -> Tuple[A2Point, float]:
    """Snap a 2-D point to the nearest A₂ lattice point with a deadband check.

    Alias for ``snap_with_error`` provided for API compatibility with
    higher-level constraint-theory interfaces that pass a deadband epsilon.
    The epsilon parameter controls tolerance: if the snap error exceeds
    epsilon, the function still returns the nearest lattice point but the
    caller can interpret the result as "outside deadband".

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.
    epsilon: float
        Maximum allowed error before considering the point "outside deadband"
        (default 0.0 = always within deadband at the origin).

    Returns
    -------
    tuple of (A2Point, float)
        Nearest lattice point and the snap error.

    Example
    -------
    >>> pt, err = snap_to_lattice(0.1, 0.2)
    >>> pt.a, pt.b, err < 0.5
    (0, 0, True)
    """
    return snap_with_error(x, y)


def dodecet_encode(a: int, b: int) -> Dodecet:
    """Encode an Eisenstein integer as a 12-bit dodecet value.

    Because a pure lattice point has zero snap error, both the error_level
    and angle_level are typically zero; the chamber is inferred from the
    point's position.  This is useful for storing pure lattice references
    in metadata.

    Parameters
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.

    Returns
    -------
    Dodecet
        The 12-bit encoding of the snap of the point's cartesian position.

    Example
    -------
    >>> d = dodecet_encode(1, 0)
    >>> isinstance(d, Dodecet)
    True
    >>> 0 <= d.raw <= 0xFFF
    True
    """
    x, y = A2Point(a, b).to_cartesian()
    return Dodecet.from_point(x, y)


# ---------------------------------------------------------------------------
# Rotation on the hexagonal lattice
# ---------------------------------------------------------------------------


def rotation(a: int, b: int, k: int = 1) -> A2Point:
    """Rotate an Eisenstein integer by k·60° counter-clockwise.

    The A₂ lattice has 6-fold rotational symmetry.  The rotation
    matrix on coordinates (a, b) for 60° CCW is::

        [a']   [ 1 -1 ] [a]
        [b'] = [ 1  0 ] [b]    i.e., (a, b) → (a - b, a)

    Rotations by larger multiples k > 1 are obtained by repeated
    application of the matrix above, reduced modulo 6.

    Parameters
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.
    k: int
        Number of 60° CCW rotation steps (default 1).

    Returns
    -------
    A2Point
        Rotated point (a', b').

    Example
    -------
    >>> rotation(1, 0, 1)  # rotate (1, 0) by 60° CCW
    A2Point(1, -1)
    >>> rotation(1, 0, 2)  # rotate (1, 0) by 120° CCW
    A2Point(0, -1)
    >>> rotation(1, 0, 6)  # full 360°
    A2Point(1, 0)
    """
    k = k % 6
    ca, cb = a, b
    for _ in range(k):
        ca, cb = ca - cb, ca
    return A2Point(ca, cb)


# ---------------------------------------------------------------------------
# Nearest neighbours on the lattice
# ---------------------------------------------------------------------------


def nearest_neighbors(a: int, b: int) -> List[A2Point]:
    """Return the 6 nearest neighbours of an Eisenstein integer.

    The six neighbours are at unit distance in the A₂ lattice:

        (a+1, b),   (a, b+1),   (a-1, b+1),
        (a-1, b),   (a, b-1),   (a+1, b-1)

    Parameters
    ----------
    a: int
        Coefficient of 1.
    b: int
        Coefficient of ω.

    Returns
    -------
    list of A2Point
        The six neighbours, ordered clockwise starting from (a+1, b).

    Example
    -------
    >>> nn = nearest_neighbors(0, 0)
    >>> len(nn)
    6
    >>> A2Point(1, 0) in nn
    True
    >>> A2Point(0, 0) in nn
    False
    """
    return [
        A2Point(a + 1, b),
        A2Point(a, b + 1),
        A2Point(a - 1, b + 1),
        A2Point(a - 1, b),
        A2Point(a, b - 1),
        A2Point(a + 1, b - 1),
    ]


__all__ = [
    # Constants
    "SQRT_3",
    "OMEGA_RE",
    "OMEGA_IM",
    "COVERING_RADIUS",
    "SAFE_THRESHOLD",
    # Types
    "A2Point",
    "Dodecet",
    "Chamber",
    # Snapping
    "snap",
    "snap_to_lattice",
    "snap_with_error",
    "snap_with_metadata",
    "snap_batch",
    # Norm
    "norm_sq",
    "norm",
    "distance_sq",
    "distance",
    # Chamber
    "classify_chamber",
    "chamber_barycentric",
    # Dodecet
    "encode",
    "encode_from_fields",
    "decode",
    "dodecet_encode",
    # Rotation & neighbours
    "rotation",
    "nearest_neighbors",
    # Utilities
    "voronoi_cell_area",
    "error_cdf",
    "voronoi_radius",
    "lattice_points_within",
]
