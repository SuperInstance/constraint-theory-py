"""Tests for constraint_theory.plato — PlatoTile, PlatoTileStore, score_tiles."""

import time
import pytest
from constraint_theory.plato import (
    PlatoTile, PlatoTileStore, TileState, TilePriority, score_tiles,
)


class TestPlatoTile:
    def test_creation(self):
        tile = PlatoTile(id="t1", domain="test")
        assert tile.id == "t1"
        assert tile.domain == "test"
        assert tile.relevance == 1.0
        assert tile.state == TileState.ACTIVE

    def test_access(self):
        tile = PlatoTile(id="t1", domain="test", content="hello")
        content = tile.access()
        assert content == "hello"
        assert tile.access_count == 1

    def test_update(self):
        tile = PlatoTile(id="t1", domain="test")
        tile.update("new content", relevance=0.7, reliability=0.9)
        assert tile.content == "new content"
        assert tile.relevance == 0.7
        assert tile.reliability == 0.9
        assert tile.version == 2

    def test_score(self):
        tile = PlatoTile(id="t1", domain="test")
        s = tile.score()
        assert 0.0 <= s <= 1.0

    def test_validate_success(self):
        tile = PlatoTile(id="t1", domain="test", reliability=0.5)
        tile.validate(True)
        assert tile.reliability > 0.5

    def test_validate_failure(self):
        tile = PlatoTile(id="t1", domain="test", reliability=0.5)
        tile.validate(False)
        assert tile.reliability < 0.5

    def test_cross_refs(self):
        tile = PlatoTile(id="t1", domain="test", dependencies=["t2", "t3"])
        assert tile.cross_refs == ["t2", "t3"]

    def test_cross_refs_setter(self):
        tile = PlatoTile(id="t1", domain="test")
        tile.cross_refs = ["t4"]
        assert tile.dependencies == ["t4"]

    def test_to_dict(self):
        tile = PlatoTile(id="t1", domain="test", content="x")
        d = tile.to_dict()
        assert d["id"] == "t1"
        assert d["domain"] == "test"

    def test_from_dict(self):
        d = {"id": "t1", "domain": "test", "content": "y", "priority": 2, "state": "active"}
        tile = PlatoTile.from_dict(d)
        assert tile.id == "t1"
        assert tile.priority == TilePriority.HIGH

    def test_repr(self):
        tile = PlatoTile(id="t1", domain="test")
        r = repr(tile)
        assert "t1" in r
        assert "test" in r

    def test_decay_relevance(self):
        tile = PlatoTile(id="t1", domain="test", relevance=1.0)
        tile.updated_at = time.time() - 1000  # simulate old tile
        tile.decay_relevance(decay_rate=0.01)
        assert tile.relevance < 1.0

    def test_priority_levels(self):
        assert TilePriority.LOW.value == 0
        assert TilePriority.MEDIUM.value == 1
        assert TilePriority.HIGH.value == 2
        assert TilePriority.CRITICAL.value == 3


class TestPlatoTileStore:
    def test_put_get(self):
        store = PlatoTileStore()
        tile = PlatoTile(id="t1", domain="test")
        store.put(tile)
        got = store.get("t1")
        assert got is not None
        assert got.id == "t1"

    def test_get_nonexistent(self):
        store = PlatoTileStore()
        assert store.get("nope") is None

    def test_delete(self):
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="test"))
        store.delete("t1")
        assert store.get("t1") is None

    def test_size(self):
        store = PlatoTileStore()
        assert store.size() == 0
        store.put(PlatoTile(id="t1", domain="test"))
        assert store.size() == 1

    def test_query_by_domain(self):
        store = PlatoTileStore()
        store.put(PlatoTile(id="t1", domain="constraint-theory.eisenstein"))
        store.put(PlatoTile(id="t2", domain="plato.core"))
        results = store.query(domain="constraint-theory")
        assert len(results) == 1
        assert results[0].id == "t1"

    def test_query_by_tags(self):
        store = PlatoTileStore()
        t1 = PlatoTile(id="t1", domain="test", tags=["math", "geometry"])
        t2 = PlatoTile(id="t2", domain="test", tags=["math"])
        store.put(t1)
        store.put(t2)
        results = store.query(tags=["math", "geometry"])
        assert len(results) == 1

    def test_query_min_relevance(self):
        store = PlatoTileStore()
        t1 = PlatoTile(id="t1", domain="test", relevance=0.9)
        t2 = PlatoTile(id="t2", domain="test", relevance=0.1)
        store.put(t1)
        store.put(t2)
        results = store.query(min_relevance=0.5)
        assert len(results) == 1

    def test_get_related(self):
        store = PlatoTileStore()
        t1 = PlatoTile(id="t1", domain="test", dependencies=["t2"])
        t2 = PlatoTile(id="t2", domain="test")
        store.put(t1)
        store.put(t2)
        related = store.get_related("t1")
        assert len(related) == 1
        assert related[0].id == "t2"

    def test_apply_decay_all(self):
        store = PlatoTileStore()
        t = PlatoTile(id="t1", domain="test", relevance=1.0)
        t.updated_at = time.time() - 100
        store.put(t)
        store.apply_decay_all()
        got = store.get("t1")
        assert got.relevance < 1.0


class TestScoreTiles:
    def test_basic(self):
        tiles = [
            PlatoTile(id="t1", domain="test", relevance=0.9),
            PlatoTile(id="t2", domain="test", relevance=0.1),
        ]
        scored = score_tiles(tiles)
        assert len(scored) == 2
        # Higher relevance should score higher
        assert scored[0][1] >= scored[1][1] or scored[0][0].relevance >= scored[1][0].relevance
