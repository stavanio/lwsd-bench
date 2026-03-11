"""
LWSD Monitor Node.

Subscribes to a state estimate topic and a ground truth / reference topic.
Computes LWSD in real time and publishes diagnostics + alerts.

Usage:
    ros2 run lwsd_bench lwsd_monitor \
        --ros-args \
        -p estimate_topic:=/odom \
        -p reference_topic:=/ground_truth/pose \
        -p control_period:=0.05 \
        -p alert_threshold:=0.5 \
        -p input_type:=odometry
"""

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64, String
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

import numpy as np

from lwsd_bench.core import LWSDComputer


class LWSDMonitor(Node):
    """
    Real-time LWSD diagnostic node.

    Subscribes to two pose sources (estimate and reference),
    computes LWSD at every reference update, publishes diagnostics.

    Supports two input types:
    - 'pose': geometry_msgs/PoseStamped on both topics
    - 'odometry': nav_msgs/Odometry for estimate, PoseStamped for reference

    The ROSMASTER M1 publishes nav_msgs/Odometry on /odom by default.
    """

    def __init__(self):
        super().__init__("lwsd_monitor")

        # --- Parameters ---
        self.declare_parameter("estimate_topic", "/odom")
        self.declare_parameter("reference_topic", "/ground_truth/pose")
        self.declare_parameter("control_period", 0.05)  # 20 Hz default
        self.declare_parameter("alert_threshold", 0.5)
        self.declare_parameter("rate_window", 10)
        self.declare_parameter("sustained_rate_cycles", 3)
        self.declare_parameter("orientation_weight", 0.0)
        self.declare_parameter("input_type", "odometry")  # 'odometry' or 'pose'
        self.declare_parameter("log_to_csv", True)
        self.declare_parameter("csv_path", "/tmp/lwsd_log.csv")

        estimate_topic = self.get_parameter("estimate_topic").value
        reference_topic = self.get_parameter("reference_topic").value
        control_period = self.get_parameter("control_period").value
        alert_threshold = self.get_parameter("alert_threshold").value
        rate_window = self.get_parameter("rate_window").value
        sustained_rate_cycles = self.get_parameter("sustained_rate_cycles").value
        orientation_weight = self.get_parameter("orientation_weight").value
        self.input_type = self.get_parameter("input_type").value
        self.log_to_csv = self.get_parameter("log_to_csv").value
        csv_path = self.get_parameter("csv_path").value

        # --- LWSD Computer ---
        self.computer = LWSDComputer(
            control_period=control_period,
            rate_window=rate_window,
            alert_threshold=alert_threshold,
            sustained_rate_cycles=sustained_rate_cycles,
            orientation_weight=orientation_weight,
        )

        # --- State ---
        self.latest_estimate = None  # (np.ndarray, float) = (state, timestamp)
        self.sample_count = 0

        # --- QoS ---
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- Subscribers ---
        if self.input_type == "odometry":
            self.create_subscription(
                Odometry, estimate_topic, self._odom_callback, sensor_qos
            )
        else:
            self.create_subscription(
                PoseStamped, estimate_topic, self._pose_estimate_callback, sensor_qos
            )

        self.create_subscription(
            PoseStamped, reference_topic, self._reference_callback, sensor_qos
        )

        # --- Publishers ---
        self.lwsd_pub = self.create_publisher(Float64, "/lwsd/value", 10)
        self.rate_pub = self.create_publisher(Float64, "/lwsd/rate", 10)
        self.latency_pub = self.create_publisher(Float64, "/lwsd/latency_ms", 10)
        self.error_pub = self.create_publisher(Float64, "/lwsd/state_error_mm", 10)
        self.alert_pub = self.create_publisher(String, "/lwsd/alert", 10)
        self.diag_pub = self.create_publisher(
            DiagnosticArray, "/diagnostics", 10
        )

        # --- CSV logging ---
        self.csv_file = None
        if self.log_to_csv:
            self.csv_file = open(csv_path, "w")
            self.csv_file.write(
                "timestamp,lwsd,lwsd_rate,state_error_m,"
                "inference_latency_ms,staleness_mm,alert\n"
            )

        self.get_logger().info(
            f"LWSD Monitor started.\n"
            f"  Estimate topic: {estimate_topic} ({self.input_type})\n"
            f"  Reference topic: {reference_topic}\n"
            f"  Control period: {control_period*1000:.1f} ms\n"
            f"  Alert threshold: {alert_threshold}"
        )

    def _odom_callback(self, msg: Odometry):
        """Cache latest odometry estimate."""
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        state = np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        stamp = self._stamp_to_sec(msg.header.stamp)
        self.latest_estimate = (state, stamp)

    def _pose_estimate_callback(self, msg: PoseStamped):
        """Cache latest PoseStamped estimate."""
        state = self._pose_to_array(msg)
        stamp = self._stamp_to_sec(msg.header.stamp)
        self.latest_estimate = (state, stamp)

    def _reference_callback(self, msg: PoseStamped):
        """
        On each reference update, compute LWSD.

        The reference message drives the computation loop. Every time
        a new ground truth / reference pose arrives, we compare it
        against the most recent estimate and compute LWSD.
        """
        if self.latest_estimate is None:
            return

        # Current time
        now = self.get_clock().now().nanoseconds * 1e-9

        # Reference state
        ref_state = self._pose_to_array(msg)

        # Estimate state and its timestamp
        est_state, est_stamp = self.latest_estimate

        # Match dimensionality: if estimate is 7D (with orientation)
        # but reference is only position, truncate
        if ref_state.shape[0] == 3 and est_state.shape[0] == 7:
            est_state = est_state[:3]

        # Inference latency: time since the estimate was produced
        inference_latency = max(now - est_stamp, 0.0)

        # Compute LWSD
        sample = self.computer.compute(
            state_true=ref_state,
            state_estimated=est_state,
            inference_latency=inference_latency,
            timestamp=now,
        )

        self.sample_count += 1

        # --- Publish ---
        self.lwsd_pub.publish(Float64(data=sample.lwsd))
        self.rate_pub.publish(Float64(data=sample.lwsd_rate))
        self.latency_pub.publish(Float64(data=sample.inference_latency * 1000.0))
        self.error_pub.publish(Float64(data=sample.staleness_mm))

        # Alert check
        alert_fired = False
        if self.computer.should_alert():
            reason = self.computer.alert_reason()
            self.alert_pub.publish(String(data=reason))
            self.get_logger().warn(f"LWSD ALERT: {reason}")
            alert_fired = True

        # Diagnostic message (every 10 samples to avoid flooding)
        if self.sample_count % 10 == 0:
            self._publish_diagnostics(sample)

        # CSV logging
        if self.csv_file:
            self.csv_file.write(
                f"{sample.timestamp:.6f},{sample.lwsd:.6f},"
                f"{sample.lwsd_rate:.6f},{sample.state_error_norm:.6f},"
                f"{sample.inference_latency*1000:.3f},{sample.staleness_mm:.3f},"
                f"{int(alert_fired)}\n"
            )

        # Console output (every 20 samples)
        if self.sample_count % 20 == 0:
            self.get_logger().info(
                f"LWSD={sample.lwsd:.4f} | "
                f"rate={sample.lwsd_rate:+.4f} | "
                f"error={sample.staleness_mm:.1f}mm | "
                f"latency={sample.inference_latency*1000:.1f}ms"
            )

    def _publish_diagnostics(self, sample):
        """Publish standard ROS2 DiagnosticArray."""
        diag = DiagnosticArray()
        diag.header.stamp = self.get_clock().now().to_msg()

        status = DiagnosticStatus()
        status.name = "LWSD Monitor"
        status.hardware_id = "perception_pipeline"

        if self.computer.should_alert():
            status.level = DiagnosticStatus.WARN
            status.message = self.computer.alert_reason()
        else:
            status.level = DiagnosticStatus.OK
            status.message = "LWSD within bounds"

        status.values = [
            KeyValue(key="lwsd", value=f"{sample.lwsd:.6f}"),
            KeyValue(key="lwsd_rate", value=f"{sample.lwsd_rate:.6f}"),
            KeyValue(key="state_error_mm", value=f"{sample.staleness_mm:.3f}"),
            KeyValue(
                key="inference_latency_ms",
                value=f"{sample.inference_latency*1000:.3f}",
            ),
            KeyValue(
                key="control_period_ms",
                value=f"{sample.control_period*1000:.3f}",
            ),
            KeyValue(key="total_samples", value=str(self.sample_count)),
        ]

        diag.status.append(status)
        self.diag_pub.publish(diag)

    def _pose_to_array(self, msg: PoseStamped) -> np.ndarray:
        """Extract position (and optionally orientation) from PoseStamped."""
        pos = msg.pose.position
        if self.computer.orientation_weight > 0:
            ori = msg.pose.orientation
            return np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        else:
            return np.array([pos.x, pos.y, pos.z])

    def _stamp_to_sec(self, stamp) -> float:
        """Convert ROS2 stamp to float seconds."""
        return stamp.sec + stamp.nanosec * 1e-9

    def destroy_node(self):
        """Clean up CSV on shutdown."""
        if self.csv_file:
            # Write summary
            summary = self.computer.summary()
            if summary:
                self.csv_file.write(f"\n# Summary\n")
                self.csv_file.write(f"# mean_lwsd: {summary.mean_lwsd:.6f}\n")
                self.csv_file.write(f"# max_lwsd: {summary.max_lwsd:.6f}\n")
                self.csv_file.write(f"# std_lwsd: {summary.std_lwsd:.6f}\n")
                self.csv_file.write(
                    f"# mean_latency_ms: {summary.mean_latency_ms:.3f}\n"
                )
                self.csv_file.write(
                    f"# max_state_error_mm: {summary.max_state_error_mm:.3f}\n"
                )
                self.csv_file.write(f"# num_alerts: {summary.num_alerts}\n")
                self.csv_file.write(f"# total_samples: {summary.total_samples}\n")
                self.csv_file.write(f"# duration_s: {summary.duration_s:.3f}\n")
                self.get_logger().info(
                    f"\n{'='*50}\n"
                    f"LWSD Session Summary\n"
                    f"{'='*50}\n"
                    f"  Duration:       {summary.duration_s:.1f}s\n"
                    f"  Samples:        {summary.total_samples}\n"
                    f"  Mean LWSD:      {summary.mean_lwsd:.4f}\n"
                    f"  Max LWSD:       {summary.max_lwsd:.4f}\n"
                    f"  Mean latency:   {summary.mean_latency_ms:.1f}ms\n"
                    f"  Max error:      {summary.max_state_error_mm:.1f}mm\n"
                    f"  Alerts:         {summary.num_alerts}\n"
                    f"{'='*50}"
                )
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LWSDMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
