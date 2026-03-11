"""
Ground Truth Publisher.

Publishes ground truth poses from a predefined set of waypoints.
Designed for the tape-mark experiment: you measure waypoint positions
in advance, then trigger each waypoint when the robot crosses it.

Two modes:
1. 'waypoints': Manual triggering via keyboard. Press Enter each time
   the robot crosses a tape mark. The node publishes the known position.
2. 'csv': Reads timestamped ground truth from a CSV file and publishes
   it at the recorded timestamps. Use this for bag replay analysis
   after extracting ground truth from video.

Usage:
    # Manual waypoint mode
    ros2 run lwsd_bench ground_truth_publisher \
        --ros-args \
        -p mode:=waypoints \
        -p waypoints_file:=/path/to/waypoints.csv

    # CSV replay mode
    ros2 run lwsd_bench ground_truth_publisher \
        --ros-args \
        -p mode:=csv \
        -p csv_file:=/path/to/ground_truth.csv
"""

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped

import csv
import threading
import sys
import time


class GroundTruthPublisher(Node):
    """
    Publishes ground truth poses for LWSD computation.

    Waypoints CSV format (one row per tape mark):
        x,y,z
        0.0,0.0,0.0
        1.0,0.0,0.0
        2.0,0.5,0.0

    Full CSV format (timestamped, from video extraction):
        timestamp,x,y,z
        0.000,0.0,0.0,0.0
        0.500,0.25,0.02,0.0
        1.000,0.50,0.03,0.0
    """

    def __init__(self):
        super().__init__("ground_truth_publisher")

        self.declare_parameter("mode", "waypoints")
        self.declare_parameter("waypoints_file", "")
        self.declare_parameter("csv_file", "")
        self.declare_parameter("output_topic", "/ground_truth/pose")
        self.declare_parameter("frame_id", "map")

        mode = self.get_parameter("mode").value
        output_topic = self.get_parameter("output_topic").value
        self.frame_id = self.get_parameter("frame_id").value

        self.pub = self.create_publisher(PoseStamped, output_topic, 10)

        if mode == "waypoints":
            waypoints_file = self.get_parameter("waypoints_file").value
            if not waypoints_file:
                self.get_logger().error("waypoints_file parameter required")
                return
            self.waypoints = self._load_waypoints(waypoints_file)
            self.current_idx = 0
            self.get_logger().info(
                f"Waypoint mode: {len(self.waypoints)} waypoints loaded.\n"
                f"Press ENTER each time the robot crosses a tape mark."
            )
            # Run keyboard listener in a thread
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_loop, daemon=True
            )
            self._keyboard_thread.start()

        elif mode == "csv":
            csv_file = self.get_parameter("csv_file").value
            if not csv_file:
                self.get_logger().error("csv_file parameter required")
                return
            self.gt_data = self._load_csv(csv_file)
            self.get_logger().info(
                f"CSV mode: {len(self.gt_data)} timestamped poses loaded.\n"
                f"Publishing at recorded timestamps."
            )
            self._csv_thread = threading.Thread(
                target=self._csv_replay_loop, daemon=True
            )
            self._csv_thread.start()

        else:
            self.get_logger().error(f"Unknown mode: {mode}")

    def _load_waypoints(self, path):
        """Load waypoints from CSV: x,y,z per line."""
        waypoints = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if len(row) >= 3:
                    waypoints.append(
                        [float(row[0]), float(row[1]), float(row[2])]
                    )
                elif len(row) >= 2:
                    waypoints.append([float(row[0]), float(row[1]), 0.0])
        return waypoints

    def _load_csv(self, path):
        """Load timestamped poses from CSV: timestamp,x,y,z per line."""
        data = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header
            for row in reader:
                if len(row) >= 4:
                    data.append(
                        {
                            "t": float(row[0]),
                            "x": float(row[1]),
                            "y": float(row[2]),
                            "z": float(row[3]),
                        }
                    )
        return data

    def _publish_pose(self, x, y, z):
        """Publish a PoseStamped with current timestamp."""
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        # Identity orientation
        msg.pose.orientation.w = 1.0
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        self.pub.publish(msg)

    def _keyboard_loop(self):
        """Wait for Enter key, publish next waypoint."""
        while self.current_idx < len(self.waypoints):
            wp = self.waypoints[self.current_idx]
            print(
                f"\n  Waypoint {self.current_idx + 1}/{len(self.waypoints)}: "
                f"({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})"
            )
            print("  Press ENTER when the robot crosses this mark...", end="")
            sys.stdout.flush()

            try:
                input()
            except EOFError:
                break

            self._publish_pose(wp[0], wp[1], wp[2])
            self.get_logger().info(
                f"Published waypoint {self.current_idx + 1}: "
                f"({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})"
            )
            self.current_idx += 1

        self.get_logger().info("All waypoints published.")

    def _csv_replay_loop(self):
        """Replay timestamped ground truth at recorded intervals."""
        if not self.gt_data:
            return

        start_time = time.monotonic()

        for i, point in enumerate(self.gt_data):
            # Wait until the relative timestamp
            target_time = start_time + point["t"]
            now = time.monotonic()
            if target_time > now:
                time.sleep(target_time - now)

            self._publish_pose(point["x"], point["y"], point["z"])

            if i % 50 == 0:
                self.get_logger().info(
                    f"Published {i+1}/{len(self.gt_data)} ground truth poses"
                )

        self.get_logger().info("CSV replay complete.")


def main(args=None):
    rclpy.init(args=args)
    node = GroundTruthPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
