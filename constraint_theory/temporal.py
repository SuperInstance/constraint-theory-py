"""
Temporal Constraints — constraint propagation with exponential time decay.

The core idea: a "deadband funnel" that shrinks over time, modelled as
epsilon decaying exponentially.  At each time step a point is snapped to
the nearest lattice point; the snap error is compared against the current
deadband width.  The funnel tightens monotonically unless an anomaly
(prediction error spike) triggers a re-widening.

References
----------
Port of the Rust temporal module from dodecet-encoder/src/temporal.rs,
adapted for pure Python with no external dependencies.

Example
-------
>>> from constraint_theory.temporal import TemporalAgent
>>> agent = TemporalAgent(decay_rate=1.0)
>>> update = agent.observe(0.1, 0.3)
>>> print(update.phase.value)
approach
"""

from __future__ import annotations

import math
from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SQRT_3: float = math.sqrt(3.0)
"""√3"""

COVERING_RADIUS: float = 1.0 / SQRT_3
"""Covering radius ρ = 1/√3 of the A₂ lattice."""

SAFE_THRESHOLD: float = COVERING_RADIUS / 2.0
"""Error below which a point is considered 'safe'."""


# ---------------------------------------------------------------------------
# Funnel phase enum
# ---------------------------------------------------------------------------

class FunnelPhase(Enum):
    """Phase of the deadband funnel at the current time step."""
    APPROACH = "approach"
    NARROWING = "narrowing"
    SNAP_IMMINENT = "snap_imminent"
    CRYSTALLIZED = "crystallized"
    ANOMALY = "anomaly"


class ChiralityState(Enum):
    """How committed the agent is to a particular Weyl chamber."""
    EXPLORING = "exploring"
    LOCKING = "locking"
    LOCKED = "locked"


class AgentAction(Enum):
    """Recommended action the agent produces after an observation."""
    CONTINUE = "continue"
    CONVERGING = "converging"
    HOLD_STEADY = "hold_steady"
    WIDEN_FUNNEL = "widen_funnel"
    COMMIT_CHIRALITY = "commit_chirality"
    DIVERGING = "diverging"
    SATISFIED = "satisfied"


# ---------------------------------------------------------------------------
# Snap result
# ---------------------------------------------------------------------------

@dataclass
class SnapResult:
    """Result of snapping a 2-D point to the nearest A₂ lattice point.

    Attributes
    ----------
    snap_a: int
        Eisenstein integer coordinate a (a + bω).
    snap_b: int
        Eisenstein integer coordinate b.
    error: float
        Euclidean distance from the input to the snap point.
    error_normalized: float
        Error divided by COVERING_RADIUS, in [0, 1].
    error_level: int
        Error quantised to 16 levels (0-15, nibble 2).
    angle_level: int
        Azimuth quantised to 16 levels (nibble 1).
    chamber: int
        Weyl chamber index (0-5).
    parity: int
        +1 for even chambers, -1 for odd.
    is_safe: bool
        True when error < SAFE_THRESHOLD.
    """
    snap_a: int
    snap_b: int
    error: float
    error_normalized: float
    error_level: int
    angle_level: int
    chamber: int
    parity: int
    is_safe: bool

    @property
    def cdf_below(self) -> float:
        """Fraction of uniformly-random points that have *smaller* error."""
        return math.pi * self.error * self.error / (SQRT_3 / 2.0)


# ---------------------------------------------------------------------------
# Eisenstein snapping (port of eisenstein.rs)
# ---------------------------------------------------------------------------

# ω = e^{2πi/3} = -1/2 + i√3/2
_OMEGA_RE: float = -0.5
_OMEGA_IM: float = SQRT_3 / 2.0

# Six Weyl chambers (sorted descending permutations of (0,1,2))
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


def snap_to_eisenstein(x: float, y: float) -> SnapResult:
    """Snap a 2-D point to the nearest Eisenstein integer (A₂ lattice).

    Uses a 9-candidate Voronoi search (guaranteed covering radius).

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.

    Returns
    -------
    SnapResult
        The nearest lattice point and associated metadata.

    Example
    -------
    >>> sr = snap_to_eisenstein(0.0, 0.0)
    >>> sr.snap_a, sr.snap_b, sr.error
    (0, 0, 0.0)
    """
    # Convert to Eisenstein coordinates
    a_f: float = x - y * _OMEGA_RE / _OMEGA_IM
    b_f: float = y / _OMEGA_IM

    a0: int = _round32(a_f)
    b0: int = _round32(b_f)

    # 9-candidate Voronoi search
    best_a: int = a0
    best_b: int = b0
    best_err: float = float("inf")

    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            ca: int = a0 + da
            cb: int = b0 + db
            cx: float = ca + cb * _OMEGA_RE
            cy: float = cb * _OMEGA_IM
            err: float = math.hypot(x - cx, y - cy)
            if err < best_err:
                best_a = ca
                best_b = cb
                best_err = err

    chamber: int = _classify_chamber(x, y)
    parity: int = 1 if chamber in _EVEN_CHAMBERS else -1

    err_norm: float = min(best_err / COVERING_RADIUS, 1.0)
    err_level: int = _round32(err_norm * 15.0)

    # Quantise angle to 16 levels (nibble 1)
    dx: float = x - (best_a + best_b * _OMEGA_RE)
    dy: float = y - (best_b * _OMEGA_IM)
    if dx != 0.0 or dy != 0.0:
        angle: float = math.atan2(dy, dx)
        norm_angle: float = (angle + math.pi) / (2.0 * math.pi)
        angle_level: int = (int(norm_angle * 16.0)) % 16
    else:
        angle_level = 0

    is_safe: bool = best_err < SAFE_THRESHOLD

    return SnapResult(
        snap_a=best_a,
        snap_b=best_b,
        error=best_err,
        error_normalized=err_norm,
        error_level=err_level,
        angle_level=angle_level,
        chamber=chamber,
        parity=parity,
        is_safe=is_safe,
    )


def _classify_chamber(x: float, y: float) -> int:
    """Classify a point into one of 6 Weyl chambers by sorting barycentric coordinates."""
    b1: float = x - y * _OMEGA_RE / _OMEGA_IM
    b2: float = y / _OMEGA_IM
    b3: float = -(b1 + b2)
    vals: List[float] = [b1, b2, b3]
    # sort indices descending by value
    sorted_idx: List[int] = sorted(range(3), key=lambda i: vals[i], reverse=True)
    perm: Tuple[int, int, int] = (sorted_idx[0], sorted_idx[1], sorted_idx[2])
    try:
        return _WEYL_PERMS.index(perm)
    except ValueError:
        return 0


def _round32(v: float) -> int:
    """Round a float to nearest int, exactly like Rust's f64.round() for 32-bit subnormals."""
    return int(math.floor(v + 0.5))


# ---------------------------------------------------------------------------
# Dodecet encoding
# ---------------------------------------------------------------------------

def encode_dodecet(sr: SnapResult) -> int:
    """Pack a SnapResult into a 12-bit dodecet value.

    Bit layout::
        bits 11-8 : error level   (nibble 2)
        bits  7-4 : angle level   (nibble 1)
        bits  3   : safety flag   (0 = safe, 1 = critical)
        bits  2-0 : chamber index (nibble 0 lower 3 bits)

    Parameters
    ----------
    sr: SnapResult
        The snap result to encode.

    Returns
    -------
    int
        12-bit value (0-4095).

    Example
    -------
    >>> sr = snap_to_eisenstein(1.5, 2.3)
    >>> dec = encode_dodecet(sr)
    >>> 0 <= dec <= 0xFFF
    True
    """
    safe_bit: int = 0 if sr.is_safe else 1
    chamber_byte: int = (safe_bit << 3) | (sr.chamber & 0x7)
    return (
        ((sr.error_level & 0xF) << 8)
        | ((sr.angle_level & 0xF) << 4)
        | (chamber_byte & 0xF)
    )


def decode_dodecet(dodecet: int) -> Tuple[int, int, int, bool]:
    """Unpack a 12-bit dodecet into its three nibbles + safety flag.

    Parameters
    ----------
    dodecet: int
        12-bit value (0-4095).

    Returns
    -------
    tuple of (error_level, angle_level, chamber, is_safe)
    """
    err_level: int = (dodecet >> 8) & 0xF
    angle_level: int = (dodecet >> 4) & 0xF
    chamber_byte: int = dodecet & 0xF
    chamber: int = chamber_byte & 0x7
    is_safe: bool = (chamber_byte >> 3) & 1 == 0
    return err_level, angle_level, chamber, is_safe


# ---------------------------------------------------------------------------
# Deadband funnel
# ---------------------------------------------------------------------------

def deadband_funnel(t: float, decay_rate: float = 1.0) -> float:
    """Compute the deadband width at normalised time *t*.

    δ(t) = ρ · (1 - t)^{1 / decay_rate}

    Parameters
    ----------
    t: float
        Normalised time in [0, 1].
    decay_rate: float
        Controls how fast the funnel narrows (default 1.0 = square-root).

    Returns
    -------
    float
        Deadband width at time *t*.
    """
    return COVERING_RADIUS * max(1.0 - t, 0.0) ** (1.0 / max(decay_rate, 0.01))


# ---------------------------------------------------------------------------
# Temporal agent
# ---------------------------------------------------------------------------

_HISTORY_SIZE: int = 64


@dataclass
class TemporalUpdate:
    """Output produced after each observation in a TemporalAgent."""

    snap: SnapResult
    """Snap result for the current observation."""
    phase: FunnelPhase
    """Current funnel phase."""
    chirality: ChiralityState
    """Current chirality state."""
    chirality_chamber: Optional[int]
    """Dominant chamber if chirality is Locking or Locked, else None."""
    predicted_error: float
    """Predicted normalised error for the next step."""
    prediction_error: float
    """Absolute difference between prediction and actual."""
    convergence_rate: float
    """Rate of change of error (negative = converging)."""
    precision_energy: float
    """Accumulated precision energy (sum of 1/error)."""
    is_anomaly: bool
    """Whether this observation triggered an anomaly."""
    action: AgentAction
    """Recommended action."""
    deadband_width: float
    """Current deadband width."""


@dataclass
class AgentSummary:
    """Snapshot of the agent's internal state for reporting."""

    history_count: int
    error_mean: float
    error_std: float
    convergence_rate: float
    precision_energy: float
    prediction_error: float
    temperature: float
    phase: FunnelPhase
    chirality: ChiralityState
    chirality_chamber: Optional[int]
    decay_rate: float
    funnel_width: float
    deadband_width: float


class TemporalAgent:
    """Constraint agent with temporal intelligence.

    The agent maintains a model of the deadband funnel over time,
    predicts future states, detects anomalies, and recommends actions.

    Parameters
    ----------
    decay_rate: float
        How fast the funnel narrows (default 1.0, range 0.1-10.0).
    prediction_horizon: int
        Steps ahead for prediction (default 4).
    anomaly_sigma: float
        Sigmas for anomaly threshold (default 2.0).
    learning_rate: float
        EMA learning rate for convergence rate (default 0.1).
    chirality_lock_threshold_milli: int
        Confidence (per-mille) to lock chirality (default 500).

    Example
    -------
    >>> agent = TemporalAgent()
    >>> u = agent.observe(0.1, 0.2)
    >>> isinstance(u, TemporalUpdate)
    True
    """

    def __init__(
        self,
        decay_rate: float = 1.0,
        prediction_horizon: int = 8,
        anomaly_sigma: float = 2.0,
        learning_rate: float = 0.1,
        chirality_lock_threshold_milli: int = 500,
    ) -> None:
        self.decay_rate = decay_rate
        self.prediction_horizon = max(1, prediction_horizon)
        self.anomaly_sigma = anomaly_sigma
        self.learning_rate = learning_rate
        self.chirality_lock_threshold_milli = chirality_lock_threshold_milli

        # Ring buffer
        self._history: List[Optional[SnapResult]] = [None] * _HISTORY_SIZE
        self._history_pos: int = 0
        self._history_count: int = 0

        # Running statistics (Welford)
        self._error_mean: float = 0.0
        self._error_var: float = 0.0

        # State
        self._convergence_rate: float = 0.0
        self._precision_energy: float = 0.0
        self._predicted_error: float = COVERING_RADIUS
        self._prediction_error: float = 0.0
        self._chirality_state: ChiralityState = ChiralityState.EXPLORING
        self._chirality_chamber: Optional[int] = None
        self._chirality_confidence: int = 0  # milliunits
        self._chirality_hops: int = 0
        self._chamber_counts: List[int] = [0] * 6
        self._phase: FunnelPhase = FunnelPhase.APPROACH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def observe(self, x: float, y: float) -> TemporalUpdate:
        """Read a sensor value and update the temporal model.

        Steps:
        1. Snap input to A₂ lattice (perception).
        2. Compare with prediction (prediction error).
        3. Update temporal model (learning).
        4. Predict next state (planning).
        5. Determine action (control).

        Parameters
        ----------
        x: float
            X coordinate.
        y: float
            Y coordinate.

        Returns
        -------
        TemporalUpdate
            Full result of this observation.
        """
        # Layer 0: Perception
        snap: SnapResult = snap_to_eisenstein(x, y)
        err_norm: float = snap.error_normalized

        # Layer 1: Control
        self._precision_energy += 1.0 / max(snap.error, 1e-300)
        self._update_convergence_rate(err_norm)

        # Layer 2: Prediction
        self._prediction_error = abs(err_norm - self._predicted_error)

        # Layer 3: Learning
        self._update_statistics(err_norm)

        # Layer 4: Planning
        self._predicted_error = self._predict_next(err_norm)

        # Chirality
        self._update_chirality(snap.chamber)

        # Phase
        self._update_phase(err_norm)

        # Store history
        self._history[self._history_pos] = snap
        self._history_pos = (self._history_pos + 1) % _HISTORY_SIZE
        self._history_count += 1

        # Anomaly detection
        err_std: float = math.sqrt(self._error_var / max(self._history_count, 1))
        is_anomaly: bool = self._prediction_error > self.anomaly_sigma * max(err_std, 0.01)

        action: AgentAction = self._determine_action(err_norm, is_anomaly)

        # Adaptive funnel
        if is_anomaly and self.decay_rate > 0.1:
            self.decay_rate *= 0.9
        elif err_norm < 0.2 and self.decay_rate < 5.0:
            self.decay_rate *= 1.05

        funnel_t: float = self.funnel_width
        db_width: float = deadband_funnel(funnel_t, self.decay_rate)

        return TemporalUpdate(
            snap=snap,
            phase=self._phase,
            chirality=self._chirality_state,
            chirality_chamber=self._chirality_chamber,
            predicted_error=self._predicted_error,
            prediction_error=self._prediction_error,
            convergence_rate=self._convergence_rate,
            precision_energy=self._precision_energy,
            is_anomaly=is_anomaly,
            action=action,
            deadband_width=db_width,
        )

    @property
    def funnel_width(self) -> float:
        """Current funnel width in [0, 1] (0 = snapped, 1 = wide-open)."""
        if self._history_count == 0:
            return 1.0
        return self._error_mean

    def summary(self) -> AgentSummary:
        """Return a snapshot of the agent's internal state."""
        err_std: float = (
            math.sqrt(self._error_var / max(self._history_count, 1))
            if self._history_count > 0
            else 0.0
        )
        return AgentSummary(
            history_count=self._history_count,
            error_mean=self._error_mean,
            error_std=err_std,
            convergence_rate=self._convergence_rate,
            precision_energy=self._precision_energy,
            prediction_error=self._prediction_error,
            temperature=self.temperature,
            phase=self._phase,
            chirality=self._chirality_state,
            chirality_chamber=self._chirality_chamber,
            decay_rate=self.decay_rate,
            funnel_width=self.funnel_width,
            deadband_width=deadband_funnel(self.funnel_width, self.decay_rate),
        )

    @property
    def temperature(self) -> float:
        """Temporal entropy — how much the agent is still exploring.

        High T (close to 1.0) = exploring many chambers.
        Low T (close to 0.0) = locked into one chamber.
        """
        total: float = float(self._history_count)
        if total < 1.0:
            return 1.0
        entropy: float = 0.0
        for c in self._chamber_counts:
            if c > 0:
                p = c / total
                entropy -= p * math.log2(p)
        max_entropy: float = math.log2(6.0)  # ≈ 2.585
        return entropy / max_entropy

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_convergence_rate(self, current: float) -> None:
        if self._history_count < 1:
            return
        prev_pos: int = (self._history_pos - 1) % _HISTORY_SIZE
        prev = self._history[prev_pos]
        if prev is not None:
            prev_norm: float = prev.error / COVERING_RADIUS
            rate: float = current - prev_norm
            self._convergence_rate = (
                self.learning_rate * rate
                + (1.0 - self.learning_rate) * self._convergence_rate
            )

    def _update_statistics(self, value: float) -> None:
        n: float = float(self._history_count + 1)
        delta: float = value - self._error_mean
        self._error_mean += delta / n
        delta2: float = value - self._error_mean
        self._error_var += delta * delta2

    def _predict_next(self, current: float) -> float:
        if self._history_count < 2:
            return current
        predicted: float = current + self._convergence_rate * self.prediction_horizon
        return max(0.0, min(predicted, 1.0))

    def _update_chirality(self, chamber: int) -> None:
        self._chamber_counts[chamber] += 1

        if self._chirality_state == ChiralityState.EXPLORING:
            self._chirality_hops += 1
            if self._chirality_hops > 10:
                dom: Optional[int] = self._dominant_chamber()
                if dom is not None:
                    conf: int = self._chamber_confidence_milli(dom)
                    if conf > self.chirality_lock_threshold_milli:
                        self._chirality_state = ChiralityState.LOCKING
                        self._chirality_chamber = dom
                        self._chirality_confidence = conf

        elif self._chirality_state == ChiralityState.LOCKING:
            if chamber == self._chirality_chamber:
                self._chirality_confidence = min(
                    self._chirality_confidence + 50, 1000
                )
                if self._chirality_confidence > 900:
                    self._chirality_state = ChiralityState.LOCKED
            else:
                self._chirality_confidence = max(
                    self._chirality_confidence - 100, 0
                )
                if self._chirality_confidence < 300:
                    self._chirality_state = ChiralityState.EXPLORING
                    self._chirality_hops = 0
                    self._chirality_chamber = None

        # LOCKED — only unlock via external anomaly signal

    def _dominant_chamber(self) -> Optional[int]:
        max_count: int = max(self._chamber_counts)
        if max_count == 0:
            return None
        for i, c in enumerate(self._chamber_counts):
            if c == max_count:
                return i
        return None

    def _chamber_confidence_milli(self, chamber: int) -> int:
        total: int = sum(self._chamber_counts)
        if total == 0:
            return 0
        return int(self._chamber_counts[chamber] / total * 1000)

    def _update_phase(self, err_norm: float) -> None:
        if err_norm > 0.9:
            self._phase = FunnelPhase.APPROACH
        elif err_norm > 0.5:
            self._phase = FunnelPhase.NARROWING
        elif err_norm > 0.15:
            self._phase = FunnelPhase.SNAP_IMMINENT
        elif err_norm < 0.05:
            self._phase = FunnelPhase.CRYSTALLIZED
        elif self._phase == FunnelPhase.ANOMALY:
            self._phase = FunnelPhase.ANOMALY
        else:
            self._phase = FunnelPhase.NARROWING

    def _determine_action(self, err_norm: float, is_anomaly: bool) -> AgentAction:
        if is_anomaly:
            return AgentAction.WIDEN_FUNNEL
        if err_norm < 0.05:
            return AgentAction.SATISFIED
        if (
            self._chirality_state == ChiralityState.LOCKED
            and self._phase != FunnelPhase.CRYSTALLIZED
        ):
            return AgentAction.COMMIT_CHIRALITY
        if self._convergence_rate < -0.01:
            return AgentAction.CONVERGING
        if self._convergence_rate > 0.01:
            return AgentAction.DIVERGING
        if err_norm < 0.2:
            return AgentAction.HOLD_STEADY
        return AgentAction.CONTINUE


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def check_constraint(x: float, y: float, funnel_width: float = 1.0) -> bool:
    """Check whether a point satisfies the A₂ lattice constraint.

    Parameters
    ----------
    x: float
        X coordinate.
    y: float
        Y coordinate.
    funnel_width: float
        Current funnel width in [0, 1] (default 1.0 = widest possible).

    Returns
    -------
    bool
        True if the snap error is within the deadband threshold.
    """
    sr: SnapResult = snap_to_eisenstein(x, y)
    threshold: float = deadband_funnel(funnel_width)
    return sr.error <= threshold


__all__ = [
    # Constants
    "SQRT_3",
    "COVERING_RADIUS",
    "SAFE_THRESHOLD",
    # Enums
    "FunnelPhase",
    "ChiralityState",
    "AgentAction",
    # Data classes
    "SnapResult",
    "TemporalUpdate",
    "AgentSummary",
    # Core functions
    "snap_to_eisenstein",
    "encode_dodecet",
    "decode_dodecet",
    "deadband_funnel",
    "check_constraint",
    # Agent
    "TemporalAgent",
]
