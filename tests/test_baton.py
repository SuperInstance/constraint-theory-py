"""Tests for constraint_theory.baton — BatonShard, split/merge/diff/validate."""

import pytest
from constraint_theory.baton import (
    BatonShard, split_context, merge_shards, diff_shards, validate_shard,
    ARTIFACTS_KEY, REASONING_KEY, BLOCKERS_KEY,
)


class TestBatonShard:
    def test_default_creation(self):
        shard = BatonShard()
        assert isinstance(shard.artifacts, dict)
        assert isinstance(shard.reasoning, list)
        assert isinstance(shard.blockers, list)
        assert isinstance(shard.metadata, dict)

    def test_with_data(self):
        shard = BatonShard(
            artifacts={"file.py": "code"},
            reasoning=["step1", "step2"],
            blockers=["bug1"],
            metadata={"version": "1.0"},
        )
        assert shard.artifacts["file.py"] == "code"
        assert len(shard.reasoning) == 2
        assert shard.blockers == ["bug1"]


class TestSplitContext:
    def test_basic_split(self):
        ctx = {
            "version": "1.0",
            "artifacts": {"file.py": "code"},
            "reasoning": ["step1"],
            "blockers": ["bug"],
        }
        shard = split_context(ctx)
        assert shard.artifacts == {"file.py": "code"}
        assert shard.reasoning == ["step1"]
        assert shard.blockers == ["bug"]
        assert shard.metadata["version"] == "1.0"

    def test_empty_context(self):
        shard = split_context({})
        assert shard.artifacts == {}
        assert shard.reasoning == []
        assert shard.blockers == []


class TestMergeShards:
    def test_roundtrip(self):
        ctx = {
            "version": "2.0",
            "artifacts": {"a.py": "x"},
            "reasoning": ["r1"],
            "blockers": ["b1"],
        }
        shard = split_context(ctx)
        merged = merge_shards(shard)
        assert merged["artifacts"] == {"a.py": "x"}
        assert merged["reasoning"] == ["r1"]
        assert merged["blockers"] == ["b1"]

    def test_empty_merge(self):
        shard = BatonShard()
        merged = merge_shards(shard)
        assert merged["artifacts"] == {}
        assert merged["reasoning"] == []
        assert merged["blockers"] == []


class TestDiffShards:
    def test_added_artifacts(self):
        left = BatonShard(artifacts={"a.py": "old"})
        right = BatonShard(artifacts={"a.py": "old", "b.py": "new"})
        diff = diff_shards(left, right)
        assert "b.py" in diff["added_artifacts"]

    def test_removed_artifacts(self):
        left = BatonShard(artifacts={"a.py": "old", "b.py": "new"})
        right = BatonShard(artifacts={"a.py": "old"})
        diff = diff_shards(left, right)
        assert "b.py" in diff["removed_artifacts"]

    def test_changed_artifacts(self):
        left = BatonShard(artifacts={"a.py": "v1"})
        right = BatonShard(artifacts={"a.py": "v2"})
        diff = diff_shards(left, right)
        assert "a.py" in diff["changed_artifacts"]

    def test_blockers_diff(self):
        left = BatonShard(blockers=["bug1"])
        right = BatonShard(blockers=["bug1", "bug2"])
        diff = diff_shards(left, right)
        assert "bug2" in diff["added_blockers"]


class TestValidateShard:
    def test_valid_shard(self):
        shard = BatonShard()
        result = validate_shard(shard)
        assert result["valid"] is True
        assert len(result["issues"]) == 0

    def test_invalid_artifacts_type(self):
        shard = BatonShard(artifacts="not a dict")
        result = validate_shard(shard)
        assert result["valid"] is False

    def test_invalid_reasoning_type(self):
        shard = BatonShard(reasoning="not a list")
        result = validate_shard(shard)
        assert result["valid"] is False

    def test_invalid_blockers_type(self):
        shard = BatonShard(blockers="not a list")
        result = validate_shard(shard)
        assert result["valid"] is False

    def test_invalid_blocker_entry(self):
        shard = BatonShard(blockers=[123])
        result = validate_shard(shard)
        assert result["valid"] is False

    def test_invalid_artifact_key(self):
        shard = BatonShard(artifacts={123: "value"})
        result = validate_shard(shard)
        assert result["valid"] is False
