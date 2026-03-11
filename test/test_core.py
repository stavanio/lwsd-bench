"""Tests for lwsd_bench.core."""

import numpy as np
import pytest
from lwsd_bench.core import LWSDComputer, LWSDSample


class TestLWSDComputer:
    """Tests for the core LWSD computation."""

    def test_basic_computation(self):
        """LWSD for known values matches hand calculation."""
        computer = LWSDComputer(control_period=0.05)  # 20 Hz

        # Object at origin, estimate 0.15m away, latency = 0.5s
        s_true = np.array([0.15, 0.0, 0.0])
        s_est = np.array([0.0, 0.0, 0.0])
        latency = 0.5

        sample = computer.compute(s_true, s_est, latency, timestamp=0.0)

        # LWSD = (0.15 / 0.05) * (0.5 / 0.05) = 3.0 * 10.0 = 30.0
        assert abs(sample.lwsd - 30.0) < 1e-6
        assert abs(sample.state_error_norm - 0.15) < 1e-6
        assert abs(sample.staleness_mm - 150.0) < 1e-6

    def test_zero_error_gives_zero_lwsd(self):
        """Perfect estimate gives LWSD = 0."""
        computer = LWSDComputer(control_period=0.05)
        s = np.array([1.0, 2.0, 3.0])
        sample = computer.compute(s, s.copy(), 0.5, timestamp=0.0)
        assert sample.lwsd == 0.0

    def test_zero_latency_gives_zero_lwsd(self):
        """Zero latency gives LWSD = 0 even with state error."""
        computer = LWSDComputer(control_period=0.05)
        s_true = np.array([1.0, 0.0, 0.0])
        s_est = np.array([0.0, 0.0, 0.0])
        sample = computer.compute(s_true, s_est, 0.0, timestamp=0.0)
        assert sample.lwsd == 0.0

    def test_monotonic_in_latency(self):
        """LWSD increases with latency for fixed error."""
        computer = LWSDComputer(control_period=0.05)
        s_true = np.array([0.1, 0.0, 0.0])
        s_est = np.array([0.0, 0.0, 0.0])

        lwsd_values = []
        for lat in [0.01, 0.05, 0.1, 0.2, 0.5]:
            computer.reset()
            sample = computer.compute(s_true, s_est, lat, timestamp=0.0)
            lwsd_values.append(sample.lwsd)

        # Should be strictly increasing
        for i in range(1, len(lwsd_values)):
            assert lwsd_values[i] > lwsd_values[i - 1]

    def test_monotonic_in_error(self):
        """LWSD increases with state error for fixed latency."""
        computer = LWSDComputer(control_period=0.05)
        latency = 0.2

        lwsd_values = []
        for err in [0.01, 0.05, 0.1, 0.2, 0.5]:
            computer.reset()
            s_true = np.array([err, 0.0, 0.0])
            s_est = np.array([0.0, 0.0, 0.0])
            sample = computer.compute(s_true, s_est, latency, timestamp=0.0)
            lwsd_values.append(sample.lwsd)

        for i in range(1, len(lwsd_values)):
            assert lwsd_values[i] > lwsd_values[i - 1]

    def test_alert_threshold(self):
        """Alert fires when LWSD exceeds threshold."""
        computer = LWSDComputer(control_period=0.05, alert_threshold=1.0)

        # Below threshold
        s_true = np.array([0.001, 0.0, 0.0])
        s_est = np.array([0.0, 0.0, 0.0])
        computer.compute(s_true, s_est, 0.05, timestamp=0.0)
        assert not computer.should_alert()

        # Above threshold
        s_true = np.array([0.5, 0.0, 0.0])
        computer.compute(s_true, s_est, 0.5, timestamp=0.1)
        assert computer.should_alert()

    def test_orientation_error(self):
        """Orientation contributes to state error when weight > 0."""
        computer_pos = LWSDComputer(
            control_period=0.05, orientation_weight=0.0
        )
        computer_full = LWSDComputer(
            control_period=0.05, orientation_weight=0.1
        )

        # Same position, different orientation
        s_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.707, 0.707])
        s_est = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        sample_pos = computer_pos.compute(s_true, s_est, 0.2, timestamp=0.0)
        sample_full = computer_full.compute(s_true, s_est, 0.2, timestamp=0.0)

        # Position-only should be 0 (positions match)
        assert sample_pos.state_error_norm < 1e-6

        # Full should be > 0 (orientations differ)
        assert sample_full.state_error_norm > 0.0

    def test_lwsd_rate(self):
        """LWSD rate is positive when divergence grows."""
        computer = LWSDComputer(control_period=0.05, rate_window=5)
        s_est = np.array([0.0, 0.0, 0.0])

        # Increasing error over time
        for i in range(10):
            err = 0.01 * (i + 1)
            s_true = np.array([err, 0.0, 0.0])
            sample = computer.compute(
                s_true, s_est, 0.2, timestamp=i * 0.05
            )

        # Rate should be positive
        assert sample.lwsd_rate > 0

    def test_summary(self):
        """Summary stats computed correctly."""
        computer = LWSDComputer(
            control_period=0.05, alert_threshold=5.0
        )
        s_est = np.array([0.0, 0.0, 0.0])

        for i in range(20):
            s_true = np.array([0.05 * (i + 1), 0.0, 0.0])
            computer.compute(s_true, s_est, 0.2, timestamp=i * 0.05)

        summary = computer.summary()
        assert summary is not None
        assert summary.total_samples == 20
        assert summary.max_lwsd > summary.mean_lwsd
        assert summary.duration_s > 0

    def test_coffee_mug_example(self):
        """
        Validates the worked example from Section 2.3 of the paper.

        Object moves at 0.3 m/s. Classical pipeline at 83 Hz (12 ms).
        VLM pipeline at 2 Hz (500 ms).
        """
        # Classical: 12 ms latency, 12 ms control period
        classical = LWSDComputer(control_period=0.012)
        # After 12 ms at 0.3 m/s: object moved 3.6 mm = 0.0036 m
        s_true = np.array([0.0036, 0.0, 0.0])
        s_est = np.array([0.0, 0.0, 0.0])
        sample_c = classical.compute(s_true, s_est, 0.012, timestamp=0.0)

        # LWSD = (0.0036 / 0.012) * (0.012 / 0.012) = 0.3
        assert abs(sample_c.lwsd - 0.3) < 0.01
        assert abs(sample_c.staleness_mm - 3.6) < 0.1

        # VLM: 500 ms latency, 500 ms control period
        vlm = LWSDComputer(control_period=0.5)
        # After 500 ms at 0.3 m/s: object moved 150 mm = 0.15 m
        s_true_v = np.array([0.15, 0.0, 0.0])
        sample_v = vlm.compute(s_true_v, s_est, 0.5, timestamp=0.0)

        # LWSD = (0.15 / 0.5) * (0.5 / 0.5) = 0.3
        assert abs(sample_v.lwsd - 0.3) < 0.01
        assert abs(sample_v.staleness_mm - 150.0) < 0.1

        # Both LWSD values are 0.3 (matching the paper's point about
        # self-referential normalization)
        assert abs(sample_c.lwsd - sample_v.lwsd) < 0.01

        # But absolute errors differ by 41.7x
        ratio = sample_v.staleness_mm / sample_c.staleness_mm
        assert abs(ratio - 41.67) < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
