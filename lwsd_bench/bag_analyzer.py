"""
Offline Bag Analyzer.

Reads a ROS2 bag, extracts state estimate and reference topics,
computes LWSD over the full trajectory, and outputs a CSV report
plus summary statistics.

Usage:
    ros2 run lwsd_bench lwsd_bag_analyzer \
        --ros-args \
        -p bag_path:=/path/to/rosbag \
        -p estimate_topic:=/odom \
        -p reference_topic:=/ground_truth/pose \
        -p control_period:=0.05 \
        -p output_csv:=/path/to/results.csv

Note: Requires rosbag2_py, which ships with standard ROS2 Humble+.
"""

import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import numpy as np
import csv
import os

from lwsd_bench.core import LWSDComputer


class BagAnalyzer(Node):
    """Offline LWSD computation from bag files."""

    def __init__(self):
        super().__init__("lwsd_bag_analyzer")

        self.declare_parameter("bag_path", "")
        self.declare_parameter("estimate_topic", "/odom")
        self.declare_parameter("reference_topic", "/ground_truth/pose")
        self.declare_parameter("control_period", 0.05)
        self.declare_parameter("alert_threshold", 0.5)
        self.declare_parameter("orientation_weight", 0.0)
        self.declare_parameter("output_csv", "/tmp/lwsd_bag_results.csv")

        bag_path = self.get_parameter("bag_path").value
        estimate_topic = self.get_parameter("estimate_topic").value
        reference_topic = self.get_parameter("reference_topic").value
        control_period = self.get_parameter("control_period").value
        alert_threshold = self.get_parameter("alert_threshold").value
        orientation_weight = self.get_parameter("orientation_weight").value
        output_csv = self.get_parameter("output_csv").value

        if not bag_path:
            self.get_logger().error("bag_path parameter is required")
            return

        self.get_logger().info(f"Analyzing bag: {bag_path}")
        self.get_logger().info(f"  Estimate topic: {estimate_topic}")
        self.get_logger().info(f"  Reference topic: {reference_topic}")

        try:
            import rosbag2_py
        except ImportError:
            self.get_logger().error(
                "rosbag2_py not found. Install with: "
                "sudo apt install ros-${ROS_DISTRO}-rosbag2-py"
            )
            return

        # Open bag
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=bag_path, storage_id="sqlite3"
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        reader.open(storage_options, converter_options)

        # Get topic type map
        topic_types = reader.get_all_topics_and_types()
        type_map = {info.name: info.type for info in topic_types}

        if estimate_topic not in type_map:
            self.get_logger().error(
                f"Topic '{estimate_topic}' not found in bag. "
                f"Available: {list(type_map.keys())}"
            )
            return
        if reference_topic not in type_map:
            self.get_logger().error(
                f"Topic '{reference_topic}' not found in bag. "
                f"Available: {list(type_map.keys())}"
            )
            return

        # Read all messages
        estimates = []  # (timestamp_ns, state_array, msg_stamp_ns)
        references = []  # (timestamp_ns, state_array)

        while reader.has_next():
            topic, data, timestamp_ns = reader.read_next()

            if topic == estimate_topic:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                state, msg_stamp = self._extract_state(
                    msg, type_map[topic], orientation_weight > 0
                )
                if state is not None:
                    estimates.append((timestamp_ns, state, msg_stamp))

            elif topic == reference_topic:
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                state, _ = self._extract_state(
                    msg, type_map[topic], orientation_weight > 0
                )
                if state is not None:
                    references.append((timestamp_ns, state))

        self.get_logger().info(
            f"Read {len(estimates)} estimates, {len(references)} references"
        )

        if not estimates or not references:
            self.get_logger().error("No data found on one or both topics")
            return

        # Compute LWSD at each reference timestamp
        computer = LWSDComputer(
            control_period=control_period,
            alert_threshold=alert_threshold,
            orientation_weight=orientation_weight,
        )

        est_idx = 0
        results = []

        for ref_ts_ns, ref_state in references:
            ref_ts = ref_ts_ns * 1e-9

            # Find most recent estimate before this reference
            while (
                est_idx < len(estimates) - 1
                and estimates[est_idx + 1][0] <= ref_ts_ns
            ):
                est_idx += 1

            if est_idx >= len(estimates):
                break

            est_ts_ns, est_state, est_msg_stamp_ns = estimates[est_idx]

            # Match dimensionality
            if ref_state.shape[0] < est_state.shape[0]:
                est_state = est_state[: ref_state.shape[0]]
            elif est_state.shape[0] < ref_state.shape[0]:
                ref_state = ref_state[: est_state.shape[0]]

            # Inference latency from message stamp
            inference_latency = max(
                (ref_ts_ns - est_msg_stamp_ns) * 1e-9, 0.0
            )

            sample = computer.compute(
                state_true=ref_state,
                state_estimated=est_state,
                inference_latency=inference_latency,
                timestamp=ref_ts,
            )
            results.append(sample)

        # Write CSV
        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "lwsd",
                    "lwsd_rate",
                    "state_error_m",
                    "inference_latency_ms",
                    "staleness_mm",
                ]
            )
            for s in results:
                writer.writerow(
                    [
                        f"{s.timestamp:.6f}",
                        f"{s.lwsd:.6f}",
                        f"{s.lwsd_rate:.6f}",
                        f"{s.state_error_norm:.6f}",
                        f"{s.inference_latency*1000:.3f}",
                        f"{s.staleness_mm:.3f}",
                    ]
                )

        self.get_logger().info(f"Results written to {output_csv}")

        # Summary
        summary = computer.summary()
        if summary:
            self.get_logger().info(
                f"\n{'='*55}\n"
                f"  LWSD Bag Analysis Summary\n"
                f"{'='*55}\n"
                f"  Duration:         {summary.duration_s:.1f} s\n"
                f"  Samples:          {summary.total_samples}\n"
                f"  Mean LWSD:        {summary.mean_lwsd:.6f}\n"
                f"  Max LWSD:         {summary.max_lwsd:.6f}\n"
                f"  Std LWSD:         {summary.std_lwsd:.6f}\n"
                f"  Mean latency:     {summary.mean_latency_ms:.1f} ms\n"
                f"  Max latency:      {summary.max_latency_ms:.1f} ms\n"
                f"  Mean error:       {summary.mean_state_error_mm:.1f} mm\n"
                f"  Max error:        {summary.max_state_error_mm:.1f} mm\n"
                f"  Alerts:           {summary.num_alerts}\n"
                f"  Sustained +rate:  {summary.num_sustained_positive_rate}\n"
                f"{'='*55}"
            )

    def _extract_state(self, msg, msg_type_str, include_orientation):
        """Extract position (and optionally orientation) from a ROS2 message."""
        msg_stamp_ns = 0

        if "Odometry" in msg_type_str:
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            msg_stamp_ns = (
                msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            )
        elif "PoseStamped" in msg_type_str:
            pos = msg.pose.position
            ori = msg.pose.orientation
            msg_stamp_ns = (
                msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
            )
        else:
            return None, 0

        if include_orientation:
            state = np.array(
                [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w]
            )
        else:
            state = np.array([pos.x, pos.y, pos.z])

        return state, msg_stamp_ns


def main(args=None):
    rclpy.init(args=args)
    node = BagAnalyzer()
    # Bag analyzer runs once and exits
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
