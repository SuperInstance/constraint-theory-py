"""
PLATO Tile Interface — a class that represents a PLATO tile with domain,
relevance, and recency scoring.

In the PLATO architecture (Persistent Long-term Associative Thought
Orchestrator), tiles are atomic units of knowledge.  Each tile has:

- A *domain* (knowledge area or namespace)
- A *relevance* score that decays over time (exponential decay)
- A *recency* timestamp (last access time)
- A *reliability* score tracking how often the tile has been validated

This module provides the ``PlatoTile`` class and scoring utilities.

Example
-------
>>> from constraint_theory.plato import PlatoTile
>>> tile = PlatoTile(id="ct.001", domain="constraint-theory", content="A₂ covering radius = 1/√3")
>>> tile.score()  # initial relevance = 1.0
1.0
>>> tile.relevance = 0.5
>>> len(repr(tile)) > 10
True
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Domain type
# ---------------------------------------------------------------------------

DomainName = str
"""A qualified domain name like ``"constraint-theory.eisenstein"``."""


# ---------------------------------------------------------------------------
# Tile state
# ---------------------------------------------------------------------------

class TileState(Enum):
    """Lifecycle state of a PLATO tile."""
    ACTIVE = "active"
    LOCKED = "locked"       # Being written; not readable
    ARCHIVED = "archived"   # Retired; no longer active
    PURGED = "purging"       # Marked for deletion


class TilePriority(Enum):
    """Priority level for tile retrieval ordering."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# PlatoTile
# ---------------------------------------------------------------------------

_DEFAULT_RELEVANCE_DECAY: float = 0.05
"""Default relevance decay rate (exponential)."""

_DEFAULT_SCORE_WEIGHTS = {
    "relevance": 0.4,
    "recency": 0.3,
    "reliability": 0.2,
    "priority": 0.1,
}
"""Default weights for composite scoring."""


@dataclass
class PlatoTile:
    """A single PLATO knowledge tile.

    Attributes
    ----------
    id: str
        Unique tile identifier (e.g. ``"ct.001"``).
    domain: str
        Knowledge domain (e.g. ``"constraint-theory.eisenstein"``).
    content: Any
        The tile's payload (str, dict, JSON-serialisable, etc.).
    relevance: float
        Relevance score in [0, 1] (decays over time).
    recency: float
        Unix timestamp of last access (monotonic).
    reliability: float
        Reliability score in [0, 1] (fraction of successful validations).
    priority: TilePriority
        Priority level (default MEDIUM).
    state: TileState
        Current lifecycle state (default ACTIVE).
    version: int
        Monotonically increasing version counter.
    tags: list of str
        Arbitrary tags for cross-domain linking.
    dependencies: list of str
        IDs of tiles this tile depends on.
    created_at: float
        Unix timestamp of creation.
    updated_at: float
        Unix timestamp of last modification.
    access_count: int
        Number of times this tile has been accessed.
    """
    id: str
    domain: DomainName
    content: Any = None
    relevance: float = 1.0
    recency: float = field(default_factory=time.time)
    reliability: float = 1.0
    priority: TilePriority = TilePriority.MEDIUM
    state: TileState = TileState.ACTIVE
    version: int = 1
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def access(self) -> Any:
        """Record an access event: update recency, increment count.

        Returns
        -------
        Any
            The tile's content.
        """
        self.recency = time.time()
        self.access_count += 1
        return self.content

    def update(
        self,
        content: Any,
        relevance: Optional[float] = None,
        reliability: Optional[float] = None,
    ) -> None:
        """Update the tile's content and optionally its scores.

        Parameters
        ----------
        content: Any
            New content payload.
        relevance: float or None
            New relevance score (if provided).
        reliability: float or None
            New reliability score (if provided).
        """
        self.content = content
        if relevance is not None:
            self.relevance = max(0.0, min(1.0, relevance))
        if reliability is not None:
            self.reliability = max(0.0, min(1.0, reliability))
        self.version += 1
        self.updated_at = time.time()

    # ------------------------------------------------------------------
    # Decay
    # ------------------------------------------------------------------

    def decay_relevance(self, decay_rate: float = _DEFAULT_RELEVANCE_DECAY) -> None:
        """Apply exponential relevance decay.

        Parameters
        ----------
        decay_rate: float
            Exponential decay rate (default 0.05).
        """
        time_since_update: float = time.time() - self.updated_at
        self.relevance = max(0.0, self.relevance * math.exp(-decay_rate * time_since_update))

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        now: Optional[float] = None,
        weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """Compute composite score from relevance, recency, reliability, priority.

        ``score = w_relevance * relevance + w_recency * recency_norm + w_reliability * reliability + w_priority * priority_norm``

        Parameters
        ----------
    now: float or None
            Current timestamp (defaults to time.time()).
        weights: dict or None
            Custom weight dict with keys ``relevance``, ``recency``,
            ``reliability``, ``priority`` (default uses ``_DEFAULT_SCORE_WEIGHTS``).

        Returns
        -------
        float
            Composite score in [0, 1].
        """
        if now is None:
            now = time.time()

        w: Dict[str, float] = weights or _DEFAULT_SCORE_WEIGHTS

        # Recency normalised to [0, 1] — exponential decay from last access
        recency_norm: float = math.exp(-0.001 * (now - self.recency))

        # Priority normalised
        priority_norm: float = self.priority.value / 3.0

        score: float = (
            w.get("relevance", 0.4) * self.relevance
            + w.get("recency", 0.3) * recency_norm
            + w.get("reliability", 0.2) * self.reliability
            + w.get("priority", 0.1) * priority_norm
        )
        return max(0.0, min(1.0, score))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, success: bool) -> None:
        """Update reliability based on a validation outcome.

        Uses an EMA-like update.

        Parameters
        ----------
        success: bool
            Whether the tile passed validation.
        """
        alpha: float = 0.1
        target: float = 1.0 if success else 0.0
        self.reliability = (1.0 - alpha) * self.reliability + alpha * target

    # ------------------------------------------------------------------
    # Cross-references (alias for ``dependencies``, per spec)
    # ------------------------------------------------------------------

    @property
    def cross_refs(self) -> List[str]:
        """Cross-references to other tiles (alias for ``dependencies``).

        This property is provided for compatibility with higher-level
        PLATO interfaces that use the ``cross_refs`` field name.
        """
        return self.dependencies

    @cross_refs.setter
    def cross_refs(self, refs: List[str]) -> None:
        self.dependencies = list(refs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "id": self.id,
            "domain": self.domain,
            "content": self.content,
            "relevance": self.relevance,
            "recency": self.recency,
            "reliability": self.reliability,
            "priority": self.priority.value,
            "state": self.state.value,
            "version": self.version,
            "tags": list(self.tags),
            "dependencies": list(self.dependencies),
            "cross_refs": list(self.dependencies),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatoTile":
        """Deserialise from a dictionary."""
        return cls(
            id=data["id"],
            domain=data["domain"],
            content=data.get("content"),
            relevance=data.get("relevance", 1.0),
            recency=data.get("recency", time.time()),
            reliability=data.get("reliability", 1.0),
            priority=TilePriority(data.get("priority", 1)),
            state=TileState(data.get("state", "active")),
            version=data.get("version", 1),
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            access_count=data.get("access_count", 0),
        )

    def __repr__(self) -> str:
        return (
            f"PlatoTile(id={self.id!r}, domain={self.domain!r}, "
            f"relv={self.relevance:.2f}, reliab={self.reliability:.2f}, "
            f"state={self.state.value}, v{self.version})"
        )


# ---------------------------------------------------------------------------
# Tile store (lightweight in-memory)
# ---------------------------------------------------------------------------


class PlatoTileStore:
    """Lightweight, memory-backed store for ``PlatoTile`` objects.

    Not suitable for production (no persistence, no indexing), but useful
    for prototyping and testing PLATO workflows.

    Parameters
    ----------
    decay_rate: float
        Global relevance decay rate (default 0.05).
    score_weights: dict or None
        Score weights for composite scoring (default ``_DEFAULT_SCORE_WEIGHTS``).

    Example
    -------
    >>> store = PlatoTileStore()
    >>> t = PlatoTile(id="t1", domain="test")
    >>> store.put(t)
    >>> got = store.get("t1")
    >>> got.id == t.id
    True
    """

    def __init__(
        self,
        decay_rate: float = _DEFAULT_RELEVANCE_DECAY,
        score_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self._tiles: Dict[str, PlatoTile] = {}
        self.decay_rate = decay_rate
        self.score_weights = score_weights or _DEFAULT_SCORE_WEIGHTS

    def put(self, tile: PlatoTile) -> None:
        """Insert or replace a tile."""
        tile.updated_at = time.time()
        self._tiles[tile.id] = tile

    def get(self, tile_id: str) -> Optional[PlatoTile]:
        """Retrieve a tile by ID.

        Automatically records an access event.
        """
        tile: Optional[PlatoTile] = self._tiles.get(tile_id)
        if tile is not None:
            tile.access()
        return tile

    def delete(self, tile_id: str) -> None:
        """Remove a tile from the store."""
        self._tiles.pop(tile_id, None)

    def query(
        self,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_relevance: float = 0.0,
        state: Optional[TileState] = None,
        limit: int = 100,
    ) -> List[PlatoTile]:
        """Query tiles by domain, tags, minimum relevance, and state.

        Results are sorted by composite score descending.

        Parameters
        ----------
        domain: str or None
            Filter by domain prefix (e.g. ``"constraint-theory"`` matches
            ``"constraint-theory.eisenstein"``).
        tags: list of str or None
            Filter by tags (tile must have all specified tags).
        min_relevance: float
            Minimum relevance threshold.
        state: TileState or None
            Filter by lifecycle state.
        limit: int
            Maximum results to return.

        Returns
        -------
        list of PlatoTile
            Matching tiles sorted by score descending.
        """
        now: float = time.time()
        results: List[PlatoTile] = []

        for tile in self._tiles.values():
            if state is not None and tile.state != state:
                continue
            if tile.relevance < min_relevance:
                continue
            if domain is not None and not tile.domain.startswith(domain):
                continue
            if tags is not None and not all(t in tile.tags for t in tags):
                continue

            tile.decay_relevance(self.decay_rate)
            results.append(tile)

        results.sort(key=lambda t: t.score(now=now, weights=self.score_weights), reverse=True)
        return results[:limit]

    def get_related(self, tile_id: str, max_depth: int = 1) -> List[PlatoTile]:
        """Retrieve tiles cross-referenced by a given tile.

        Follows the tile's ``cross_refs`` (``dependencies``) up to
        ``max_depth`` hops and returns all uniquely-referenced tiles.

        Parameters
        ----------
        tile_id: str
            ID of the source tile.
        max_depth: int
            Maximum number of hops to follow (default 1).

        Returns
        -------
        list of PlatoTile
            Related tiles (does not include the source tile itself).
        """
        tile: Optional[PlatoTile] = self._tiles.get(tile_id)
        if tile is None:
            return []

        seen: set = {tile_id}
        related: List[PlatoTile] = []
        frontier: List[str] = list(tile.cross_refs)

        for _depth in range(max_depth):
            if not frontier:
                break
            next_frontier: List[str] = []
            for ref_id in frontier:
                if ref_id in seen:
                    continue
                seen.add(ref_id)
                ref_tile: Optional[PlatoTile] = self._tiles.get(ref_id)
                if ref_tile is not None:
                    related.append(ref_tile)
                    next_frontier.extend(ref_tile.cross_refs)
            frontier = [r for r in next_frontier if r not in seen]

        return related

    def size(self) -> int:
        """Number of tiles currently in the store."""
        return len(self._tiles)

    def apply_decay_all(self) -> None:
        """Apply relevance decay to every tile in the store."""
        for tile in self._tiles.values():
            tile.decay_relevance(self.decay_rate)


# ---------------------------------------------------------------------------
# Utility: batch scoring
# ---------------------------------------------------------------------------


def score_tiles(
    tiles: Sequence[PlatoTile],
    now: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
) -> List[Tuple[PlatoTile, float]]:
    """Compute composite scores for multiple tiles.

    Parameters
    ----------
    tiles: sequence of PlatoTile
        Tiles to score.
    now: float or None
        Current timestamp.
    weights: dict or None
        Custom score weights.

    Returns
    -------
    list of (PlatoTile, float)
        Tiles with their scores.
    """
    if now is None:
        now = time.time()
    return [(t, t.score(now=now, weights=weights)) for t in tiles]


__all__ = [
    # Types
    "DomainName",
    "TileState",
    "TilePriority",
    # Core
    "PlatoTile",
    "PlatoTileStore",
    # Utility
    "score_tiles",
]
