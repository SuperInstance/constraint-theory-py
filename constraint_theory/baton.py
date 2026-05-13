"""
Baton Shard — split a context dictionary into three shards: artifacts,
reasoning, and blockers.

In the Cocapn fleet protocol, agents carry a "baton" (a shared context dict)
that is passed between team members.  The baton is split into three shards
for efficient serialisation, handoff, and targeted updates:

- **artifacts**: Files, code, output data (large, immutable blobs).
- **reasoning**: Chain-of-thought, plans, decisions (moderate, append-heavy).
- **blockers**: Active problems, open questions, stalled items (small, priority).

Shards can be re-assembled, merged, or diffed after remote updates.

Example
-------
>>> from constraint_theory.baton import BatonShard, split_context, merge_shards
>>> ctx = {"version": "1.0", "artifacts": {"file.py": "print(1)"}, "reasoning": ["step1"], "blockers": ["bug"]}
>>> shards = split_context(ctx)
>>> shards.artifacts["file.py"]
'print(1)'
>>> merge_shards(shards)["version"]
'1.0'
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Shard keys
# ---------------------------------------------------------------------------

ARTIFACTS_KEY: str = "artifacts"
"""Key in the context dict that holds artifacts."""

REASONING_KEY: str = "reasoning"
"""Key in the context dict that holds reasoning traces."""

BLOCKERS_KEY: str = "blockers"
"""Key in the context dict that holds active blockers."""

_SHARD_KEYS: Tuple[str, str, str] = (ARTIFACTS_KEY, REASONING_KEY, BLOCKERS_KEY)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

Artifacts = Dict[str, Any]
"""Artifact shard: a mapping of names to blob contents."""

Reasoning = List[Any]
"""Reasoning shard: a list of step entries."""

Blockers = List[str]
"""Blockers shard: a list of problem descriptions."""

ShardDict = Dict[str, Any]
"""A single shard represented as a generic dict for JSON safety."""


# ---------------------------------------------------------------------------
# BatonShard
# ---------------------------------------------------------------------------


@dataclass
class BatonShard:
    """Three-way baton context split.

    Attributes
    ----------
    artifacts: dict of str -> Any
        Files, code, output data (large blobs).
    reasoning: list
        Step-by-step chain of thought, plans, decisions.
    blockers: list of str
        Active problems and open questions.
    metadata: dict
        Optional metadata (timestamps, agent ID, round number, etc.).
    """

    artifacts: Artifacts = field(default_factory=dict)
    reasoning: Reasoning = field(default_factory=list)
    blockers: Blockers = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_artifact(self, name: str, content: Any) -> None:
        """Add or replace an artifact.

        Parameters
        ----------
        name: str
            Artifact name (e.g. ``"src/main.py"``).
        content: Any
            Artifact content.
        """
        self.artifacts[name] = content

    def add_reasoning(self, *steps: Any) -> None:
        """Append one or more reasoning steps.

        Parameters
        ----------
        *steps: Any
            Steps to append.
        """
        self.reasoning.extend(steps)

    def add_blocker(self, blocker: str) -> None:
        """Add a blocker description.

        Parameters
        ----------
        blocker: str
            Description of the blocker.
        """
        if blocker not in self.blockers:
            self.blockers.append(blocker)

    def resolve_blocker(self, blocker: str) -> None:
        """Remove a blocker that has been resolved.

        Parameters
        ----------
        blocker: str
            Blocker to remove.
        """
        self.blockers = [b for b in self.blockers if b != blocker]

    def has_blockers(self) -> bool:
        """True if there is at least one active blocker."""
        return len(self.blockers) > 0

    def artifact_count(self) -> int:
        """Number of artifacts."""
        return len(self.artifacts)

    def reasoning_count(self) -> int:
        """Number of reasoning steps."""
        return len(self.reasoning)

    def blocker_count(self) -> int:
        """Number of blockers."""
        return len(self.blockers)

    def artifact_hash(self, name: str) -> Optional[str]:
        """SHA-256 hex digest of a single artifact's JSON repr.

        Parameters
        ----------
        name: str
            Artifact name.

        Returns
        -------
        str or None
            Hex digest if the artifact exists.
        """
        content: Optional[Any] = self.artifacts.get(name)
        if content is None:
            return None
        raw: bytes = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def integrity(self) -> str:
        """SHA-256 root hash of the entire shard set (content only).

        Returns
        -------
        str
            Hex digest of joined shard hashes.
        """
        digests: List[str] = []
        for shard_name in ("artifacts", "reasoning", "blockers", "metadata"):
            raw: bytes = json.dumps(
                getattr(self, shard_name), sort_keys=True, default=str
            ).encode("utf-8")
            digests.append(hashlib.sha256(raw).hexdigest())
        root: bytes = "|".join(digests).encode("utf-8")
        return hashlib.sha256(root).hexdigest()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a flat dict."""
        return {
            ARTIFACTS_KEY: dict(self.artifacts),
            REASONING_KEY: list(self.reasoning),
            BLOCKERS_KEY: list(self.blockers),
            "metadata": dict(self.metadata),
        }

    def to_json(self, **kwargs: Any) -> str:
        """Serialise to a JSON string.

        Parameters
        ----------
        **kwargs: Any
            Passed through to ``json.dumps`` (e.g. ``indent=2``).

        Returns
        -------
        str
            JSON representation.
        """
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatonShard":
        """Deserialise from a dict.

        Parameters
        ----------
        data: dict
            A dict with optional keys ``artifacts``, ``reasoning``,
            ``blockers``, ``metadata``.

        Returns
        -------
        BatonShard
        """
        return cls(
            artifacts=data.get(ARTIFACTS_KEY, {}),
            reasoning=data.get(REASONING_KEY, []),
            blockers=data.get(BLOCKERS_KEY, []),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "BatonShard":
        """Deserialise from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __repr__(self) -> str:
        return (
            f"BatonShard(artifacts={self.artifact_count()}, "
            f"reasoning={self.reasoning_count()}, "
            f"blockers={self.blocker_count()}, "
            f"hash={self.integrity()[:16]}...)"
        )


# ---------------------------------------------------------------------------
# Split / merge utilities
# ---------------------------------------------------------------------------


def split_context(context: Dict[str, Any]) -> BatonShard:
    """Split a flat context dict into a three-way ``BatonShard``.

    The three shard keys (``artifacts``, ``reasoning``, ``blockers``) are
    extracted from the context and placed into the corresponding shard
    attributes.  All other context keys are placed into ``metadata``.

    Parameters
    ----------
    context: dict
        The full baton context.

    Returns
    -------
    BatonShard
        The split shard.

    Example
    -------
    >>> ctx = {"version": "1.0", "artifacts": {"x": 1}, "blockers": ["bug"]}
    >>> s = split_context(ctx)
    >>> s.artifacts["x"]
    1
    >>> s.blockers
    ['bug']
    """
    shard = BatonShard()
    for key, value in context.items():
        if key == ARTIFACTS_KEY:
            shard.artifacts = value if isinstance(value, dict) else {}
        elif key == REASONING_KEY:
            shard.reasoning = value if isinstance(value, list) else []
        elif key == BLOCKERS_KEY:
            shard.blockers = value if isinstance(value, list) else []
        else:
            shard.metadata[key] = value
    return shard


def merge_shards(shard: BatonShard, extra_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Reassemble a ``BatonShard`` back into a flat context dict.

    ``metadata`` from the shard and any ``extra_metadata`` are merged
    (extra takes precedence).

    Parameters
    ----------
    shard: BatonShard
        The split shard.
    extra_metadata: dict or None
        Additional metadata to include in the merged context.

    Returns
    -------
    dict
        Reassembled context dict with keys ``artifacts``, ``reasoning``,
        ``blockers``, plus all metadata entries.
    """
    context: Dict[str, Any] = {
        ARTIFACTS_KEY: dict(shard.artifacts),
        REASONING_KEY: list(shard.reasoning),
        BLOCKERS_KEY: list(shard.blockers),
    }
    # Merge metadata
    metadata: Dict[str, Any] = {}
    metadata.update(shard.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    context.update(metadata)
    return context


# ---------------------------------------------------------------------------
# Diff / merge utilities
# ---------------------------------------------------------------------------


def diff_shards(left: BatonShard, right: BatonShard) -> Dict[str, Any]:
    """Compute a shallow structural diff between two shard instances.

    Returns a dict with keys ``added_artifacts``, ``removed_artifacts``,
    ``changed_artifacts``, ``added_blockers``, ``removed_blockers``,
    ``reasoning_left_count``, ``reasoning_right_count``.

    Parameters
    ----------
    left: BatonShard
        First (older) shard.
    right: BatonShard
        Second (newer) shard.

    Returns
    -------
    dict
        Diff summary.
    """
    left_arts: set = set(left.artifacts.keys())
    right_arts: set = set(right.artifacts.keys())

    added: set = right_arts - left_arts
    removed: set = left_arts - right_arts
    common: set = left_arts & right_arts
    changed: set = {n for n in common if left.artifacts[n] != right.artifacts[n]}

    left_blockers: set = set(left.blockers)
    right_blockers: set = set(right.blockers)

    return {
        "added_artifacts": sorted(added),
        "removed_artifacts": sorted(removed),
        "changed_artifacts": sorted(changed),
        "added_blockers": sorted(right_blockers - left_blockers),
        "removed_blockers": sorted(left_blockers - right_blockers),
        "reasoning_left_count": len(left.reasoning),
        "reasoning_right_count": len(right.reasoning),
    }


# ---------------------------------------------------------------------------
# Module __all__
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shard validation
# ---------------------------------------------------------------------------


def validate_shard(shard: BatonShard) -> Dict[str, Any]:
    """Check a ``BatonShard`` for completeness and structural validity.

    Returns a dict with keys:

    * ``valid`` (bool) — True if all checks pass.
    * ``issues`` (list of str) — Descriptions of each problem found.

    Checks performed:

    * ``artifacts`` must be a dict.
    * ``reasoning`` must be a list.
    * ``blockers`` must be a list.
    * ``metadata`` must be a dict.
    * ``artifacts`` keys must be strings.
    * ``blockers`` entries must be strings.
    * If there are blocking issues, ``valid`` is False and ``issues``
      lists each problem.

    Parameters
    ----------
    shard: BatonShard
        The shard to validate.

    Returns
    -------
    dict
        ``{"valid": True/False, "issues": [str, ...]}``

    Example
    -------
    >>> s = BatonShard()
    >>> result = validate_shard(s)
    >>> result["valid"]
    True
    >>> len(result["issues"])
    0
    """
    issues: List[str] = []

    if not isinstance(shard.artifacts, dict):
        issues.append(f"artifacts must be a dict, got {type(shard.artifacts).__name__}")
    else:
        for key in shard.artifacts:
            if not isinstance(key, str):
                issues.append(f"artifact key must be str, got {type(key).__name__}: {key!r}")

    if not isinstance(shard.reasoning, list):
        issues.append(f"reasoning must be a list, got {type(shard.reasoning).__name__}")

    if not isinstance(shard.blockers, list):
        issues.append(f"blockers must be a list, got {type(shard.blockers).__name__}")
    else:
        for b in shard.blockers:
            if not isinstance(b, str):
                issues.append(f"blocker must be str, got {type(b).__name__}: {b!r}")

    if not isinstance(shard.metadata, dict):
        issues.append(f"metadata must be a dict, got {type(shard.metadata).__name__}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }


__all__ = [
    "ARTIFACTS_KEY",
    "REASONING_KEY",
    "BLOCKERS_KEY",
    "BatonShard",
    "split_context",
    "merge_shards",
    "diff_shards",
    "validate_shard",
]
