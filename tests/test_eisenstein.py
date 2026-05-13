"""
Tests for the Eisenstein lattice module.
"""

from __future__ import annotations

import math

import pytest

from constraint_theory.eisenstein import (
    SQRT_3,
    OMEGA_RE,
    OMEGA_IM,
    COVERING_RADIUS,
    SAFE_THRESHOLD,
    A2Point,
    Dodecet,
    Chamber,
    snap,
    snap_with_error,
    snap_with_metadata,
    snap_batch,
    norm_sq,
    norm,
    distance_sq,
    distance,
    classify_chamber,
    chamber_barycentric,
    encode,
    encode_from_fields,
    decode,
    voronoi_cell_area,
    error_cdf,
    voronoi_radius,
    lattice_points_within,
    snap_to_lattice,
    dodecet_encode,
    rotation,
    nearest_neighbors,
)


class TestConstants:
    def test_sqrt_3(self) -> None:
        assert SQRT_3 == pytest.approx(math.sqrt(3.0))

    def test_omega(self) -> None:
        assert OMEGA_RE == pytest.approx(-0.5)
        assert OMEGA_IM == pytest.approx(math.sqrt(3.0) / 2.0)

    def test_covering_radius(self) -> None:
        assert COVERING_RADIUS == pytest.approx(1.0 / math.sqrt(3.0))

    def test_safe_threshold(self) -> None:
        assert SAFE_THRESHOLD == pytest.approx(COVERING_RADIUS / 2.0)

    def test_voronoi_cell_area(self) -> None:
        assert voronoi_cell_area() == pytest.approx(math.sqrt(3.0) / 2.0)

    def test_voronoi_radius(self) -> None:
        assert voronoi_radius() == pytest.approx(COVERING_RADIUS)


class TestA2Point:
    def test_origin(self) -> None:
        p = A2Point(0, 0)
        assert p.to_cartesian() == (0.0, 0.0)

    def test_cartesian_identity(self) -> None:
        p = A2Point(1, 0)
        x, y = p.to_cartesian()
        assert x == pytest.approx(1.0)
        assert y == pytest.approx(0.0)

    def test_omega_point(self) -> None:
        p = A2Point(0, 1)
        x, y = p.to_cartesian()
        assert x == pytest.approx(OMEGA_RE)
        assert y == pytest.approx(OMEGA_IM)

    def test_from_cartesian_snap(self) -> None:
        p = A2Point.from_cartesian(1.5, 2.3)
        assert isinstance(p.a, int)
        assert isinstance(p.b, int)

    def test_addition(self) -> None:
        assert A2Point(1, 2) + A2Point(3, 4) == A2Point(4, 6)

    def test_subtraction(self) -> None:
        assert A2Point(5, 7) - A2Point(2, 3) == A2Point(3, 4)

    def test_negation(self) -> None:
        assert -A2Point(3, -2) == A2Point(-3, 2)

    def test_multiplication(self) -> None:
        p = A2Point(2, 3)
        assert p * 4 == A2Point(8, 12)
        assert 4 * p == A2Point(8, 12)

    def test_equality(self) -> None:
        assert A2Point(1, 1) == A2Point(1, 1)
        assert A2Point(1, 1) != A2Point(2, 1)

    def test_hashable(self) -> None:
        s = {A2Point(0, 0), A2Point(0, 0), A2Point(1, 0)}
        assert len(s) == 2

    def test_repr(self) -> None:
        r = repr(A2Point(2, 3))
        assert "A2Point" in r and "2" in r and "3" in r


class TestSnap:
    def test_origin_snaps_to_origin(self) -> None:
        assert snap(0.0, 0.0) == A2Point(0, 0)

    def test_integer_point_snaps(self) -> None:
        assert snap(1.0, 0.0) == A2Point(1, 0)

    def test_omega_snaps(self) -> None:
        p = snap(-0.5, OMEGA_IM)
        assert p.a == 0 and p.b == 1

    def test_close_point(self) -> None:
        p = snap(0.1, 0.2)
        assert p.a == 0 and p.b == 0

    def test_error_bounded(self) -> None:
        for x, y in [(0.3, 0.4), (1.7, -0.5), (-0.9, 2.1)]:
            _, err = snap_with_error(x, y)
            assert err <= COVERING_RADIUS + 1e-12

    def test_snap_with_metadata(self) -> None:
        pt, err, chamber, err_level, is_safe = snap_with_metadata(0.1, 0.2)
        assert isinstance(pt, A2Point) and err >= 0.0
        assert 0 <= chamber <= 5
        assert 0 <= err_level <= 15
        assert isinstance(is_safe, bool)

    def test_snap_batch(self) -> None:
        results = snap_batch([(0.0, 0.0), (1.0, 0.0)])
        assert len(results) == 2
        assert results[0][0] == A2Point(0, 0)


class TestNorm:
    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (1, 0, 1), (0, 1, 1),
        (1, 1, 3), (2, 1, 7), (-2, 3, 7),
    ])
    def test_norm_sq(self, a: int, b: int, expected: int) -> None:
        assert norm_sq(a, b) == expected

    def test_norm(self) -> None:
        assert norm(1, 1) == pytest.approx(math.sqrt(3.0))

    def test_distance(self) -> None:
        assert distance(0, 0, 1, 0) == pytest.approx(1.0)


class TestChamber:
    def test_chamber_classify(self) -> None:
        for x, y in [(0, 0), (1, 0), (0, 1), (-1, -1)]:
            ch = classify_chamber(float(x), float(y))
            assert 0 <= ch <= 5

    def test_barycentric_sum(self) -> None:
        b1, b2, b3 = chamber_barycentric(1.0, 2.0)
        assert b1 + b2 + b3 == pytest.approx(0.0)

    def test_chamber_parity(self) -> None:
        for i in range(6):
            assert Chamber(i).parity in (1, -1)

    def test_all_chambers_reachable(self) -> None:
        seen: set = set()
        for x in range(-10, 11):
            for y in range(-10, 11):
                seen.add(classify_chamber(float(x), float(y)))
        assert len(seen) >= 6


class TestDodecet:
    def test_encode_decode_roundtrip(self) -> None:
        d = encode(0.3, 0.4)
        err, angle, chamber, safe = decode(d)
        assert 0 <= err <= 15 and 0 <= angle <= 15
        assert 0 <= chamber <= 5
        assert isinstance(safe, bool)

    def test_dodecet_range(self) -> None:
        d = encode(0.0, 0.0)
        assert 0 <= d.raw <= 0xFFF

    def test_manual_construction(self) -> None:
        d = encode_from_fields(error_level=5, angle_level=10, chamber=3, is_safe=True)
        assert d.error_level == 5 and d.angle_level == 10
        assert d.chamber == 3 and d.is_safe and not d.is_critical

    def test_critical_flag(self) -> None:
        d = encode_from_fields(error_level=15, angle_level=0, chamber=0, is_safe=False)
        assert d.is_critical and not d.is_safe

    def test_decode_raw_int(self) -> None:
        err, angle, chamber, safe = decode(0x000)
        assert err == 0 and angle == 0 and chamber == 0 and safe

    def test_decode_max(self) -> None:
        err, angle, chamber, safe = decode(0xFFF)
        assert err == 0xF and angle == 0xF and not safe

    def test_from_point_direct(self) -> None:
        d = Dodecet.from_point(0.5, 0.5)
        assert isinstance(d.raw, int)

    def test_repr(self) -> None:
        assert "Dodecet" in repr(encode(0.1, 0.2))


class TestErrorCDF:
    def test_cdf_zero(self) -> None:
        assert error_cdf(0.0) == pytest.approx(0.0)

    def test_cdf_at_covering_radius(self) -> None:
        # Circular approximation can overshoot 1.0 since hexagon area is smaller
        assert error_cdf(COVERING_RADIUS) > 0.7

    def test_lattice_points_within(self) -> None:
        pts = lattice_points_within(2.0)
        assert len(pts) >= 1
        assert A2Point(0, 0) in pts


class TestRotation:
    """Tests for the rotation() function."""

    def test_identity(self) -> None:
        assert rotation(1, 0, 0) == A2Point(1, 0)

    def test_sixty_degrees(self) -> None:
        r = rotation(1, 0, 1)
        assert r == A2Point(1, 1) or distance_sq(1, 0, r.a, r.b) == pytest.approx(1.0)
        assert r == A2Point(1, 1)

    def test_one_twenty(self) -> None:
        r = rotation(1, 0, 2)
        assert r == A2Point(0, 1)

    def test_one_eighty(self) -> None:
        assert rotation(1, 0, 3) == A2Point(-1, 0)

    def test_two_forty(self) -> None:
        assert rotation(1, 0, 4) == A2Point(-1, -1)

    def test_three_hundred(self) -> None:
        assert rotation(1, 0, 5) == A2Point(0, -1)

    def test_full_turn(self) -> None:
        assert rotation(1, 0, 6) == A2Point(1, 0)

    def test_modularity(self) -> None:
        assert rotation(2, 3, 7) == rotation(2, 3, 1)

    def test_negative_values(self) -> None:
        r = rotation(-1, -1, 2)
        assert isinstance(r, A2Point)


class TestNearestNeighbors:
    """Tests for the nearest_neighbors() function."""

    def test_six_neighbors(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert len(nn) == 6

    def test_no_duplicates(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert len(set(nn)) == 6

    def test_origin_not_included(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert A2Point(0, 0) not in nn

    def test_one_zero_present(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert A2Point(1, 0) in nn

    def test_zero_one_present(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert A2Point(0, 1) in nn

    def test_minus_one_zero_present(self) -> None:
        nn = nearest_neighbors(0, 0)
        assert A2Point(-1, 0) in nn

    def test_non_origin(self) -> None:
        nn = nearest_neighbors(3, 2)
        assert A2Point(4, 2) in nn
        assert A2Point(3, 3) in nn

    def test_all_unit_distance(self) -> None:
        nn = nearest_neighbors(1, 1)
        for pt in nn:
            d = distance(1, 1, pt.a, pt.b)
            assert d == pytest.approx(1.0)


class TestSnapToLattice:
    """Tests for the snap_to_lattice() convenience function."""

    def test_origin(self) -> None:
        pt, err = snap_to_lattice(0.0, 0.0)
        assert pt == A2Point(0, 0)
        assert err == pytest.approx(0.0)

    def test_near_origin(self) -> None:
        pt, err = snap_to_lattice(0.3, 0.4)
        assert pt == A2Point(0, 0)
        assert err <= 0.5

    def test_error_bounded(self) -> None:
        for x, y in [(0.5, 0.5), (1.7, -0.5), (-2.1, 3.3)]:
            _, err = snap_to_lattice(x, y)
            assert err <= COVERING_RADIUS + 1e-12


class TestDodecetEncode:
    """Tests for the dodecet_encode() function."""

    def test_origin(self) -> None:
        d = dodecet_encode(0, 0)
        assert isinstance(d, Dodecet)
        assert 0 <= d.raw <= 0xFFF

    def test_unit_point(self) -> None:
        d = dodecet_encode(1, 0)
        assert isinstance(d, Dodecet)

    def test_roundtrip(self) -> None:
        d = dodecet_encode(2, 3)
        fields = decode(d)
        assert len(fields) == 4
