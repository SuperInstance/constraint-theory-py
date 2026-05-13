"""
constraint-theory v0.3.0 — Pure Python constraint satisfaction toolkit.

Features in v0.3.0:

  1. **Temporal constraints** — exponential-decay deadband funnel with
     anomaly detection, chirality locking, and phase tracking.
  2. **Eisenstein lattice** — A₂ lattice snap, Weyl chamber classification,
     dodecet (12-bit) encoding, norm/rotation/neighbour operations.
  3. **Adaptive tolerance** — ε(c) = k/c for manifold boundary regions,
     region classification, curvature estimation.
  4. **PLATO tile interface** — domain/relevance/recency/reliability scoring,
     tile store with search by domain, tags, and state.
  5. **Baton shard** — split context into artifacts/reasoning/blockers,
     merge, diff, and validate.

All modules are pure Python with full type hints and no external dependencies
beyond the standard library.
"""

from constraint_theory.version import __version__

# ---------------------------------------------------------------------------
# Re-export the most important names for ergonomic ``from constraint_theory import X``
# ---------------------------------------------------------------------------

from constraint_theory.temporal import (
    COVERING_RADIUS,
    SAFE_THRESHOLD,
    SQRT_3,
    FunnelPhase,
    ChiralityState,
    AgentAction,
    SnapResult,
    TemporalUpdate,
    AgentSummary,
    TemporalAgent,
    snap_to_eisenstein,
    encode_dodecet,
    decode_dodecet,
    deadband_funnel,
    check_constraint,
)

from constraint_theory.eisenstein import (
    A2Point,
    Dodecet,
    Chamber,
    snap,
    snap_to_lattice,
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
    dodecet_encode,
    rotation,
    nearest_neighbors,
    voronoi_cell_area,
    error_cdf,
    voronoi_radius,
    lattice_points_within,
)

from constraint_theory.adaptive import (
    ManifoldRegion,
    AdaptiveTolerance,
    adaptive_epsilon,
    classify_region,
    curvature_from_manifold,
    manifold_distance,
    adaptive_snap,
)

from constraint_theory.plato import (
    PlatoTile,
    PlatoTileStore,
    TileState,
    TilePriority,
    score_tiles,
)

from constraint_theory.baton import (
    BatonShard,
    split_context,
    merge_shards,
    diff_shards,
    validate_shard,
)

__all__ = [
    "__version__",
    # Temporal
    "COVERING_RADIUS",
    "SAFE_THRESHOLD",
    "SQRT_3",
    "FunnelPhase",
    "ChiralityState",
    "AgentAction",
    "SnapResult",
    "TemporalUpdate",
    "AgentSummary",
    "TemporalAgent",
    "snap_to_eisenstein",
    "encode_dodecet",
    "decode_dodecet",
    "deadband_funnel",
    "check_constraint",
    # Eisenstein
    "A2Point",
    "Dodecet",
    "Chamber",
    "snap",
    "snap_to_lattice",
    "snap_with_error",
    "snap_with_metadata",
    "snap_batch",
    "norm_sq",
    "norm",
    "distance_sq",
    "distance",
    "classify_chamber",
    "chamber_barycentric",
    "encode",
    "encode_from_fields",
    "decode",
    "dodecet_encode",
    "rotation",
    "nearest_neighbors",
    "voronoi_cell_area",
    "error_cdf",
    "voronoi_radius",
    "lattice_points_within",
    # Adaptive
    "ManifoldRegion",
    "AdaptiveTolerance",
    "adaptive_epsilon",
    "classify_region",
    "curvature_from_manifold",
    "manifold_distance",
    "adaptive_snap",
    # PLATO
    "PlatoTile",
    "PlatoTileStore",
    "TileState",
    "TilePriority",
    "score_tiles",
    # Baton
    "BatonShard",
    "split_context",
    "merge_shards",
    "diff_shards",
    "validate_shard",
]
