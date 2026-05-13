"""
Tests for temporal constraints, adaptive tolerance, PLATO tiles, and baton shards.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import pytest

# ============================================================================
# Temporal module tests
# ============================================================================

from constraint_theory.temporal import (
    TemporalAgent,
    TemporalUpdate,
    AgentSummary,
    SnapResult,
    FunnelPhase,
    ChiralityState,
    AgentAction,
    snap_to_eisenstein,
    encode_dodecet,
    decode_dodecet,
    deadband_funnel,
    check_constraint,
    COVERING_RADIUS as TEMP_COVERING,
)


class TestTemporalSnap:
    def test_snap_result_has_fields(self) -> None:
        sr = snap_to_eisenstein(0.3, 0.4)
        assert isinstance(sr.snap_a, int)
        assert isinstance(sr.snap_b, int)
        assert sr.error >= 0.0
        assert 0 <= sr.error_level <= 15
        assert 0 <= sr.angle_level <= 15
        assert 0 <= sr.chamber <= 5
        assert sr.parity in (1, -1)
        assert isinstance(sr.is_safe, bool)
        assert sr.cdf_below > 0.0

    def test_snap_zero(self) -> None:
        sr = snap_to_eisenstein(0.0, 0.0)
        assert sr.snap_a == 0 and sr.snap_b == 0
        assert sr.error == 0.0 and sr.error_level == 0
        assert sr.is_safe

    def test_snap_error_bounded(self) -> None:
        sr = snap_to_eisenstein(0.7, 0.3)
        assert sr.error <= TEMP_COVERING + 1e-12


class TestTemporalAgent:
    def test_default_construction(self) -> None:
        agent = TemporalAgent()
        assert agent.decay_rate == pytest.approx(1.0)
        assert agent.prediction_horizon == 8

    def test_custom_construction(self) -> None:
        agent = TemporalAgent(decay_rate=2.0, prediction_horizon=4, learning_rate=0.05)
        assert agent.decay_rate == pytest.approx(2.0)
        assert agent.prediction_horizon == 4
        assert agent.learning_rate == pytest.approx(0.05)

    def test_first_observation(self) -> None:
        agent = TemporalAgent()
        u = agent.observe(0.1, 0.2)
        assert isinstance(u, TemporalUpdate)
        assert isinstance(u.snap, SnapResult)
        assert isinstance(u.phase, FunnelPhase)
        assert isinstance(u.chirality, ChiralityState)
        assert isinstance(u.action, AgentAction)

    def test_multiple_observations(self) -> None:
        agent = TemporalAgent()
        for i in range(10):
            x, y = math.sin(i * 0.5), math.cos(i * 0.5)
            u = agent.observe(x, y)
            assert u.precision_energy > 0.0

    def test_summary(self) -> None:
        agent = TemporalAgent()
        agent.observe(0.1, 0.2)
        agent.observe(0.3, 0.4)
        s = agent.summary()
        assert isinstance(s, AgentSummary)
        assert s.history_count == 2 and s.error_mean > 0.0

    def test_temperature(self) -> None:
        agent = TemporalAgent()
        assert agent.temperature == pytest.approx(1.0)
        for _ in range(100):
            agent.observe(0.1, 0.1)

    def test_funnel_width_range(self) -> None:
        agent = TemporalAgent()
        assert 0.0 <= agent.funnel_width <= 1.0

    def test_check_constraint(self) -> None:
        assert check_constraint(0.0, 0.0)
        assert check_constraint(0.0, 0.0, funnel_width=0.0)

    def test_deadband_funnel_limits(self) -> None:
        assert deadband_funnel(0.0) == pytest.approx(TEMP_COVERING)
        assert deadband_funnel(1.0) == pytest.approx(0.0)

    def test_anomaly_detection(self) -> None:
        agent = TemporalAgent()
        for _ in range(10):
            agent.observe(0.01, 0.01)
        u = agent.observe(1.0, 1.0)
        assert u.prediction_error > 0.01 or u.is_anomaly

    def test_encode_decode_dodecet(self) -> None:
        sr = snap_to_eisenstein(0.3, 0.4)
        value = encode_dodecet(sr)
        err, angle, chamber, safe = decode_dodecet(value)
        assert err == sr.error_level and angle == sr.angle_level
        assert chamber == sr.chamber and safe == sr.is_safe

    def test_update_all_fields(self) -> None:
        u = TemporalAgent().observe(0.5, 0.5)
        for attr in ("snap", "phase", "chirality", "predicted_error",
                     "prediction_error", "convergence_rate", "precision_energy",
                     "is_anomaly", "action", "deadband_width"):
            assert hasattr(u, attr)

    def test_chirality_evolution(self) -> None:
        agent = TemporalAgent(chirality_lock_threshold_milli=200)
        for _ in range(50):
            agent.observe(0.1, 0.1)
        s = agent.summary()
        assert s.chirality in (ChiralityState.EXPLORING, ChiralityState.LOCKING, ChiralityState.LOCKED)


# ============================================================================
# Adaptive tolerance tests
# ============================================================================

from constraint_theory.adaptive import (
    AdaptiveTolerance,
    ManifoldRegion,
    adaptive_epsilon,
    classify_region,
    curvature_from_manifold,
    manifold_distance,
    adaptive_snap,
    DEFAULT_EPSILON_MAX,
)


class TestAdaptiveEpsilon:
    def test_inverse_proportionality(self) -> None:
        assert adaptive_epsilon(100.0) < adaptive_epsilon(10.0)

    def test_min_clamp(self) -> None:
        eps = adaptive_epsilon(1e15, k=1.0, epsilon_min=1e-12)
        assert eps >= 1e-12

    def test_max_clamp(self) -> None:
        assert adaptive_epsilon(0.0) <= DEFAULT_EPSILON_MAX

    def test_near_singularity(self) -> None:
        assert adaptive_epsilon(1e6, k=0.5) == pytest.approx(5e-7)

    def test_custom_constants(self) -> None:
        assert adaptive_epsilon(5.0, k=2.0, epsilon_max=10.0) == pytest.approx(0.4)

    def test_invalid_negative_curvature(self) -> None:
        with pytest.raises(ValueError):
            adaptive_epsilon(-1.0)

    def test_invalid_nan(self) -> None:
        with pytest.raises(ValueError):
            adaptive_epsilon(float("nan"))


class TestRegionClassification:
    def test_far(self) -> None:
        assert classify_region(0.001) == ManifoldRegion.FAR

    def test_approaching(self) -> None:
        assert classify_region(0.1) == ManifoldRegion.APPROACHING

    def test_near(self) -> None:
        assert classify_region(5.0) == ManifoldRegion.NEAR

    def test_critical(self) -> None:
        assert classify_region(100.0) == ManifoldRegion.CRITICAL

    def test_singular(self) -> None:
        assert classify_region(float("inf")) == ManifoldRegion.SINGULAR


class TestAdaptiveToleranceClass:
    def test_call(self) -> None:
        assert AdaptiveTolerance()(100.0) == pytest.approx(0.01)

    def test_batch(self) -> None:
        results = AdaptiveTolerance().batch([1.0, 10.0, 100.0])
        assert len(results) == 3
        assert all(0.0 < r <= DEFAULT_EPSILON_MAX for r in results)

    def test_caching(self) -> None:
        at = AdaptiveTolerance()
        _ = at(42.0), at(42.0)
        stats = at.cache_stats()
        assert stats["hits"] >= 1 and stats["misses"] >= 1

    def test_clear_cache(self) -> None:
        at = AdaptiveTolerance()
        _ = at(42.0)
        at.clear_cache()
        assert at.cache_stats()["size"] == 0

    def test_curvature_estimation(self) -> None:
        eps = curvature_from_manifold(0.0, 0.0, lambda x, y: 1.0)
        assert eps == pytest.approx(0.0, abs=1e-6)


class TestManifoldDistance:
    """Tests for the manifold_distance() convenience function."""

    def test_to_single_point(self) -> None:
        d = manifold_distance(0.0, 0.0, [(1.0, 0.0)])
        assert d == pytest.approx(1.0)

    def test_to_nearest(self) -> None:
        d = manifold_distance(0.0, 0.0, [(3.0, 0.0), (0.0, 4.0)])
        assert d == pytest.approx(3.0)

    def test_identical(self) -> None:
        d = manifold_distance(2.0, 3.0, [(2.0, 3.0), (10.0, 10.0)])
        assert d == pytest.approx(0.0)

    def test_many_points(self) -> None:
        pts = [(float(i), float(i)) for i in range(20)]
        d = manifold_distance(5.0, 5.0, pts)
        assert d == pytest.approx(0.0)

    def test_empty_boundary(self) -> None:
        d = manifold_distance(1.0, 1.0, [])
        assert d == float("inf")


class TestAdaptiveSnap:
    """Tests for the adaptive_snap() convenience function."""

    def test_adapts_epsilon(self) -> None:
        eps, dist, ok = adaptive_snap(0.0, 0.0, [(1.0, 0.0)])
        assert eps > 0.0
        assert dist == pytest.approx(1.0)
        assert isinstance(ok, bool)

    def test_at_origin_near_boundary(self) -> None:
        eps, dist, ok = adaptive_snap(0.0, 0.0, [(0.0, 0.0)])
        # distance zero -> curvature infinite -> eps should be epsilon_min
        assert eps == pytest.approx(1e-12, abs=1e-11)

    def test_far_from_boundary(self) -> None:
        eps, dist, ok = adaptive_snap(0.0, 0.0, [(100.0, 100.0)])
        # distance ~141 -> curvature small -> eps should be epsilon_max
        assert eps == pytest.approx(0.5)

    def test_custom_k_proportionality(self) -> None:
        # k doubles epsilon at same curvature: eps = k * curvature^{-1}
        # Both clamped to max, so compare against DEFAULT_EPSILON_MAX
        eps1, _, _ = adaptive_snap(0.0, 0.0, [(5.0, 0.0)], k=1.0)
        eps2, _, _ = adaptive_snap(0.0, 0.0, [(5.0, 0.0)], k=2.0)
        # Both hit the max clamp of 0.5 at this distance
        assert eps1 == eps2 == 0.5


# ============================================================================
# PLATO tile tests
# ============================================================================

from constraint_theory.plato import (
    PlatoTile,
    PlatoTileStore,
    TileState,
    TilePriority,
    score_tiles,
)


class TestPlatoTile:
    def test_minimal_construction(self) -> None:
        t = PlatoTile(id="ct.001", domain="constraint-theory")
        assert t.id == "ct.001" and t.relevance == pytest.approx(1.0)
        assert t.reliability == pytest.approx(1.0) and t.state == TileState.ACTIVE
        assert t.version == 1

    def test_full_construction(self) -> None:
        t = PlatoTile(id="ct.002", domain="ct.eisenstein",
                       content={"snap": {"a": 1, "b": 0}},
                       relevance=0.8, priority=TilePriority.HIGH,
                       tags=["eisenstein"], dependencies=["ct.001"])
        assert t.priority == TilePriority.HIGH and "eisenstein" in t.tags

    def test_access(self) -> None:
        t = PlatoTile(id="t1", domain="test", content="hello")
        assert t.access() == "hello" and t.access_count == 1

    def test_update(self) -> None:
        t = PlatoTile(id="t1", domain="test")
        t.update(content="updated", relevance=0.5, reliability=0.9)
        assert t.content == "updated" and t.relevance == pytest.approx(0.5)
        assert t.reliability == pytest.approx(0.9) and t.version == 2

    def test_score_range(self) -> None:
        assert 0.0 <= PlatoTile(id="t1", domain="test").score() <= 1.0

    def test_decay_relevance(self) -> None:
        t = PlatoTile(id="t1", domain="test")
        t.decay_relevance(decay_rate=0.1)
        assert t.relevance < 1.0

    def test_validate_success(self) -> None:
        t = PlatoTile(id="t1", domain="test", reliability=0.5)
        t.validate(success=True)
        assert t.reliability > 0.5

    def test_validate_failure(self) -> None:
        t = PlatoTile(id="t1", domain="test", reliability=0.5)
        t.validate(success=False)
        assert t.reliability < 0.5

    def test_serialization_roundtrip(self) -> None:
        t = PlatoTile(id="ct.003", domain="test", content={"k": "v"},
                       tags=["a", "b"])
        restored = PlatoTile.from_dict(t.to_dict())
        assert restored.id == t.id and restored.content == t.content

    def test_repr(self) -> None:
        assert "ct.001" in repr(PlatoTile(id="ct.001", domain="test"))

    def test_score_tiles_utility(self) -> None:
        tiles = [PlatoTile(id="a", domain="d", relevance=1.0),
                 PlatoTile(id="b", domain="d", relevance=0.5)]
        scored = score_tiles(tiles)
        assert len(scored) == 2
        assert all(isinstance(t, PlatoTile) and isinstance(s, float) for t, s in scored)


class TestPlatoTileStore:
    def test_put_and_get(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test"))
        assert store.get("t1") is not None

    def test_get_nonexistent(self) -> None:
        assert PlatoTileStore().get("nope") is None

    def test_delete(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test"))
        store.delete("t1")
        assert store.get("t1") is None

    def test_query_by_domain(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="ct.eisenstein"))
        store.put(PlatoTile(id="t2", domain="plato.tile"))
        assert len(store.query(domain="ct")) == 1

    def test_query_by_tags(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test", tags=["a", "b"]))
        store.put(PlatoTile(id="t2", domain="test", tags=["a"]))
        assert len(store.query(tags=["a", "b"])) == 1

    def test_query_min_relevance(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test", relevance=0.1))
        store.put(PlatoTile(id="t2", domain="test", relevance=0.9))
        assert len(store.query(min_relevance=0.5)) == 1

    def test_query_limit(self) -> None:
        store = PlatoTileStore()
        for i in range(10):
            store.put(PlatoTile(id=f"t{i}", domain="test"))
        assert len(store.query(limit=3)) == 3

    def test_apply_decay_all(self) -> None:
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test", relevance=0.5))
        store.apply_decay_all()
        t = store.get("t1")
        assert t is not None and t.relevance <= 0.5

    def test_size(self) -> None:
        store = PlatoTileStore()
        assert store.size() == 0
        store.put(PlatoTile(id="t1", domain="test"))
        assert store.size() == 1


class TestPlatoCrossRefs:
    """Tests for cross_references and get_related."""

    def test_cross_refs_property(self) -> None:
        t = PlatoTile(id="a", domain="test", dependencies=["b", "c"])
        assert t.cross_refs == ["b", "c"]

    def test_cross_refs_setter(self) -> None:
        t = PlatoTile(id="a", domain="test")
        t.cross_refs = ["x", "y"]
        assert t.dependencies == ["x", "y"]

    def test_get_related_empty(self) -> None:
        store = PlatoTileStore()
        assert store.get_related("nonexistent") == []

    def test_get_related_direct(self) -> None:
        store = PlatoTileStore()
        t1 = PlatoTile(id="t1", domain="test")
        t2 = PlatoTile(id="t2", domain="test")
        t3 = PlatoTile(id="t3", domain="test")
        t1.cross_refs = ["t2", "t3"]
        store.put(t1)
        store.put(t2)
        store.put(t3)
        related = store.get_related("t1")
        assert len(related) == 2
        assert any(t.id == "t2" for t in related)
        assert any(t.id == "t3" for t in related)

    def test_get_related_depth(self) -> None:
        store = PlatoTileStore()
        t1 = PlatoTile(id="t1", domain="test", dependencies=["t2"])
        t2 = PlatoTile(id="t2", domain="test", dependencies=["t3"])
        t3 = PlatoTile(id="t3", domain="test")
        store.put(t1); store.put(t2); store.put(t3)
        # depth 1 -> only t2
        assert len(store.get_related("t1", max_depth=1)) == 1
        # depth 2 -> t2 and t3
        assert len(store.get_related("t1", max_depth=2)) == 2

    def test_to_dict_includes_cross_refs(self) -> None:
        t = PlatoTile(id="t1", domain="test", dependencies=["t2"])
        d = t.to_dict()
        assert "cross_refs" in d
        assert d["cross_refs"] == ["t2"]


# ============================================================================
# Baton shard tests
# ============================================================================

from constraint_theory.baton import (
    BatonShard,
    split_context,
    merge_shards,
    diff_shards,
    validate_shard,
    ARTIFACTS_KEY,
    REASONING_KEY,
    BLOCKERS_KEY,
)


class TestBatonShard:
    def test_empty(self) -> None:
        s = BatonShard()
        assert s.artifact_count() == 0 and s.reasoning_count() == 0
        assert s.blocker_count() == 0 and not s.has_blockers()

    def test_add_artifact(self) -> None:
        s = BatonShard()
        s.add_artifact("file.py", "print(1)")
        assert s.artifact_count() == 1

    def test_add_reasoning(self) -> None:
        s = BatonShard()
        s.add_reasoning("step1", "step2")
        assert s.reasoning_count() == 2

    def test_add_blocker(self) -> None:
        s = BatonShard()
        s.add_blocker("bug")
        assert s.has_blockers() and s.blocker_count() == 1

    def test_no_duplicate_blockers(self) -> None:
        s = BatonShard()
        s.add_blocker("b")
        s.add_blocker("b")
        assert s.blocker_count() == 1

    def test_resolve_blocker(self) -> None:
        s = BatonShard()
        s.add_blocker("b")
        s.resolve_blocker("b")
        assert not s.has_blockers()

    def test_integrity_hash(self) -> None:
        s = BatonShard()
        h = s.integrity()
        assert len(h) == 64
        s.add_artifact("x", 1)
        assert s.integrity() != h

    def test_artifact_hash(self) -> None:
        s = BatonShard()
        s.add_artifact("x", "hello")
        assert s.artifact_hash("x") is not None
        assert s.artifact_hash("missing") is None

    def test_serialization_roundtrip(self) -> None:
        s = BatonShard(artifacts={"f": "code"}, reasoning=["r"],
                        blockers=["b"], metadata={"agent": "f"})
        restored = BatonShard.from_dict(s.to_dict())
        assert restored.artifacts == s.artifacts
        assert restored.reasoning == s.reasoning
        assert restored.blockers == s.blockers

    def test_json_roundtrip(self) -> None:
        s = BatonShard(artifacts={"a": 1})
        restored = BatonShard.from_json(s.to_json())
        assert restored.artifacts == s.artifacts

    def test_repr(self) -> None:
        assert "BatonShard" in repr(BatonShard(artifacts={"a": 1}))


class TestSplitMerge:
    def test_split_context(self) -> None:
        ctx: Dict[str, Any] = {
            "version": "1.0",
            ARTIFACTS_KEY: {"f.py": "code"},
            REASONING_KEY: ["step1"],
            BLOCKERS_KEY: ["bug"],
        }
        s = split_context(ctx)
        assert s.artifacts["f.py"] == "code"
        assert s.reasoning == ["step1"]
        assert s.blockers == ["bug"]
        assert s.metadata["version"] == "1.0"

    def test_split_empty(self) -> None:
        s = split_context({})
        assert s.artifact_count() == 0 and s.metadata == {}

    def test_merge_shards(self) -> None:
        s = BatonShard(artifacts={"f1": "data"}, reasoning=["r1"],
                        blockers=["b1"], metadata={"ts": 123})
        ctx = merge_shards(s, extra_metadata={"agent": "fm"})
        assert ctx[ARTIFACTS_KEY] == {"f1": "data"}
        assert ctx["ts"] == 123
        assert ctx["agent"] == "fm"

    def test_merge_extra_overrides(self) -> None:
        s = BatonShard(metadata={"agent": "old"})
        ctx = merge_shards(s, extra_metadata={"agent": "new"})
        assert ctx["agent"] == "new"


class TestDiffShards:
    def test_identical(self) -> None:
        l = BatonShard(artifacts={"a": 1}, blockers=["b"])
        r = BatonShard(artifacts={"a": 1}, blockers=["b"])
        d = diff_shards(l, r)
        assert d["added_artifacts"] == []
        assert d["removed_artifacts"] == []

    def test_added_artifact(self) -> None:
        d = diff_shards(BatonShard(artifacts={"a": 1}),
                        BatonShard(artifacts={"a": 1, "b": 2}))
        assert d["added_artifacts"] == ["b"]

    def test_removed_artifact(self) -> None:
        d = diff_shards(BatonShard(artifacts={"a": 1, "b": 2}),
                        BatonShard(artifacts={"a": 1}))
        assert d["removed_artifacts"] == ["b"]

    def test_changed_artifact(self) -> None:
        d = diff_shards(BatonShard(artifacts={"a": 1}),
                        BatonShard(artifacts={"a": 99}))
        assert d["changed_artifacts"] == ["a"]

    def test_blocker_changes(self) -> None:
        d = diff_shards(BatonShard(blockers=["b1", "b2"]),
                        BatonShard(blockers=["b2", "b3"]))
        assert "b1" in d["removed_blockers"]
        assert "b3" in d["added_blockers"]
        assert "b2" not in d["added_blockers"]


class TestValidateShard:
    """Tests for the validate_shard function."""

    def test_empty_valid(self) -> None:
        result = validate_shard(BatonShard())
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_valid_full(self) -> None:
        s = BatonShard(
            artifacts={"f": "code"},
            reasoning=["step1", "step2"],
            blockers=["bug", "todo"],
            metadata={"agent": "fm"},
        )
        result = validate_shard(s)
        assert result["valid"] is True

    def test_invalid_artifacts_type(self) -> None:
        s = BatonShard()
        s.artifacts = [1, 2, 3]  # type: ignore[assignment]
        result = validate_shard(s)
        assert result["valid"] is False
        assert any("artifacts" in issue for issue in result["issues"])

    def test_invalid_reasoning_type(self) -> None:
        s = BatonShard()
        s.reasoning = "not a list"  # type: ignore[assignment]
        result = validate_shard(s)
        assert result["valid"] is False

    def test_invalid_blockers_type(self) -> None:
        s = BatonShard()
        s.blockers = "not a list"  # type: ignore[assignment]
        result = validate_shard(s)
        assert result["valid"] is False

    def test_non_string_blocker(self) -> None:
        s = BatonShard(blockers=["ok", 42])  # type: ignore[list-item]
        result = validate_shard(s)
        assert result["valid"] is False
        assert any("42" in issue for issue in result["issues"])

    def test_non_string_artifact_key(self) -> None:
        s = BatonShard()
        s.artifacts = {1: "val"}  # type: ignore[dict-item]
        result = validate_shard(s)
        assert result["valid"] is False

    def test_invalid_metadata_type(self) -> None:
        s = BatonShard()
        s.metadata = "not a dict"  # type: ignore[assignment]
        result = validate_shard(s)
        assert result["valid"] is False

    def test_multiple_issues(self) -> None:
        s = BatonShard()
        s.artifacts = "bad"  # type: ignore[assignment]
        s.blockers = "worse"  # type: ignore[assignment]
        result = validate_shard(s)
        assert result["valid"] is False
        assert len(result["issues"]) >= 2
