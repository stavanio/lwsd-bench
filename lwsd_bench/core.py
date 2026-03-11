"""
Core LWSD computation.

Pure NumPy, no ROS dependency. This module can be used standalone
for offline analysis or imported by the ROS2 node.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from collections import deque


@dataclass
class LWSDSample:
    """Single LWSD measurement."""

    timestamp: float  # seconds
    lwsd: float  # dimensionless (when normalized) or m/s
    lwsd_rate: float  # d(LWSD)/dt
    state_error_norm: float  # meters
    inference_latency: float  # seconds
    control_period: float  # seconds
    staleness_mm: float  # absolute staleness in millimeters


@dataclass
class LWSDSummary:
    """Summary statistics over a trajectory."""

    mean_lwsd: float
    max_lwsd: float
    std_lwsd: float
    mean_latency_ms: float
    max_latency_ms: float
    mean_state_error_mm: float
    max_state_error_mm: float
    num_alerts: int  # times LWSD exceeded threshold
    num_sustained_positive_rate: int  # sustained positive LWSD rate episodes
    total_samples: int
    duration_s: float


class LWSDComputer:
    """
    Computes Latency-Weighted State Divergence.

    LWSD(t) = (||s(t) - s_hat(t)||_2 / tau_ctrl) * (tau_infer(t) / tau_ctrl)

    Parameters
    ----------
    control_period : float
        Control cycle period in seconds. For the ROSMASTER M1 running
        its default navigation stack, this is typically 0.05 (20 Hz).
    rate_window : int
        Number of samples for LWSD rate estimation via linear regression.
    alert_threshold : float or None
        LWSD value above which an alert is triggered. None disables alerts.
    sustained_rate_cycles : int
        Number of consecutive positive-rate cycles before a sustained
        rate alert fires.
    orientation_weight : float
        Weight (in meters/radian) for orientation error when computing
        state error on SE(3) states. Set to 0 for position-only tracking.
    """

    def __init__(
        self,
        control_period: float = 0.05,
        rate_window: int = 10,
        alert_threshold: Optional[float] = None,
        sustained_rate_cycles: int = 3,
        orientation_weight: float = 0.0,
    ):
        self.control_period = control_period
        self.rate_window = rate_window
        self.alert_threshold = alert_threshold
        self.sustained_rate_cycles = sustained_rate_cycles
        self.orientation_weight = orientation_weight

        # Ring buffer for rate estimation
        self._buffer: deque = deque(maxlen=rate_window)
        self._time_buffer: deque = deque(maxlen=rate_window)

        # Sustained positive rate counter
        self._positive_rate_count = 0

        # History for summary
        self._history: list[LWSDSample] = []

    def compute(
        self,
        state_true: np.ndarray,
        state_estimated: np.ndarray,
        inference_latency: float,
        timestamp: float,
    ) -> LWSDSample:
        """
        Compute LWSD for a single timestep.

        Parameters
        ----------
        state_true : np.ndarray
            Ground truth state. Shape (3,) for position [x,y,z] or
            (7,) for pose [x,y,z,qx,qy,qz,qw].
        state_estimated : np.ndarray
            Robot's state estimate, same shape as state_true.
        inference_latency : float
            Time in seconds between sensor capture and estimate availability.
            Computed as (current_time - estimate_timestamp).
        timestamp : float
            Current time in seconds.

        Returns
        -------
        LWSDSample
            LWSD value, rate, and supporting diagnostics.
        """
        # State error
        state_error = self._state_error(state_true, state_estimated)

        # LWSD: (error / tau_ctrl) * (tau_infer / tau_ctrl)
        lwsd = (state_error / self.control_period) * (
            inference_latency / self.control_period
        )

        # Update ring buffer for rate estimation
        self._buffer.append(lwsd)
        self._time_buffer.append(timestamp)

        # LWSD rate via linear regression over window
        lwsd_rate = self._compute_rate()

        # Sustained positive rate tracking
        if lwsd_rate > 0:
            self._positive_rate_count += 1
        else:
            self._positive_rate_count = 0

        sample = LWSDSample(
            timestamp=timestamp,
            lwsd=lwsd,
            lwsd_rate=lwsd_rate,
            state_error_norm=state_error,
            inference_latency=inference_latency,
            control_period=self.control_period,
            staleness_mm=state_error * 1000.0,
        )

        self._history.append(sample)
        return sample

    def should_alert(self) -> bool:
        """Check if current state warrants a safety alert."""
        if not self._history:
            return False

        latest = self._history[-1]

        # Threshold-based alert
        if self.alert_threshold is not None and latest.lwsd > self.alert_threshold:
            return True

        # Sustained positive rate alert
        if self._positive_rate_count >= self.sustained_rate_cycles:
            return True

        return False

    def alert_reason(self) -> str:
        """Return human-readable reason for alert, or empty string."""
        if not self._history:
            return ""

        latest = self._history[-1]
        reasons = []

        if self.alert_threshold is not None and latest.lwsd > self.alert_threshold:
            reasons.append(
                f"LWSD={latest.lwsd:.4f} > threshold={self.alert_threshold:.4f}"
            )

        if self._positive_rate_count >= self.sustained_rate_cycles:
            reasons.append(
                f"Sustained positive LWSD rate for "
                f"{self._positive_rate_count} cycles"
            )

        return "; ".join(reasons)

    def summary(self) -> Optional[LWSDSummary]:
        """Compute summary statistics over all recorded samples."""
        if not self._history:
            return None

        lwsd_vals = np.array([s.lwsd for s in self._history])
        latencies = np.array([s.inference_latency for s in self._history])
        errors = np.array([s.state_error_norm for s in self._history])

        # Count alerts
        num_alerts = 0
        if self.alert_threshold is not None:
            num_alerts = int(np.sum(lwsd_vals > self.alert_threshold))

        # Count sustained positive rate episodes
        num_sustained = 0
        consecutive = 0
        for s in self._history:
            if s.lwsd_rate > 0:
                consecutive += 1
                if consecutive == self.sustained_rate_cycles:
                    num_sustained += 1
            else:
                consecutive = 0

        duration = self._history[-1].timestamp - self._history[0].timestamp

        return LWSDSummary(
            mean_lwsd=float(np.mean(lwsd_vals)),
            max_lwsd=float(np.max(lwsd_vals)),
            std_lwsd=float(np.std(lwsd_vals)),
            mean_latency_ms=float(np.mean(latencies)) * 1000.0,
            max_latency_ms=float(np.max(latencies)) * 1000.0,
            mean_state_error_mm=float(np.mean(errors)) * 1000.0,
            max_state_error_mm=float(np.max(errors)) * 1000.0,
            num_alerts=num_alerts,
            num_sustained_positive_rate=num_sustained,
            total_samples=len(self._history),
            duration_s=duration,
        )

    def reset(self):
        """Clear all history and buffers."""
        self._buffer.clear()
        self._time_buffer.clear()
        self._positive_rate_count = 0
        self._history.clear()

    def _state_error(
        self, s_true: np.ndarray, s_est: np.ndarray
    ) -> float:
        """
        Compute state error norm.

        For position-only (shape 3): Euclidean distance.
        For pose (shape 7): position + weighted orientation geodesic.
        """
        if s_true.shape[0] == 3:
            # Position only
            return float(np.linalg.norm(s_true - s_est))

        elif s_true.shape[0] >= 7:
            # Position + quaternion [x,y,z,qx,qy,qz,qw]
            pos_err = np.linalg.norm(s_true[:3] - s_est[:3])

            if self.orientation_weight > 0:
                q1 = s_true[3:7] / np.linalg.norm(s_true[3:7])
                q2 = s_est[3:7] / np.linalg.norm(s_est[3:7])

                # Geodesic distance on SO(3)
                dot = np.abs(np.clip(np.dot(q1, q2), -1.0, 1.0))
                angle_err = 2.0 * np.arccos(dot)

                return float(
                    np.sqrt(
                        pos_err**2
                        + (self.orientation_weight * angle_err) ** 2
                    )
                )
            else:
                return float(pos_err)

        else:
            # Generic: Euclidean
            return float(np.linalg.norm(s_true - s_est))

    def _compute_rate(self) -> float:
        """Linear regression slope over the ring buffer."""
        n = len(self._buffer)
        if n < 2:
            return 0.0

        t = np.array(self._time_buffer)
        y = np.array(self._buffer)

        # Normalize time to avoid numerical issues
        t = t - t[0]

        if t[-1] - t[0] < 1e-9:
            return 0.0

        # Slope via least squares: slope = cov(t,y) / var(t)
        t_mean = np.mean(t)
        y_mean = np.mean(y)
        numerator = np.sum((t - t_mean) * (y - y_mean))
        denominator = np.sum((t - t_mean) ** 2)

        if abs(denominator) < 1e-12:
            return 0.0

        return float(numerator / denominator)
