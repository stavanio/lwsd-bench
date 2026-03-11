"""
Latency Injector Node.

Subscribes to a pose/odometry topic, delays it by a configurable amount,
and republishes. This is the key tool for the controlled experiment:
same robot, same environment, different perception latencies.

Usage:
    # Native latency (baseline)
    ros2 run lwsd_bench latency_injector \
        --ros-args \
        -p input_topic:=/odom \
        -p output_topic:=/odom_delayed \
        -p delay_ms:=0.0

    # +200 ms injected latency
    ros2 run lwsd_bench latency_injector \
        --ros-args \
        -p input_topic:=/odom \
        -p output_topic:=/odom_delayed \
        -p delay_ms:=200.0

    # +500 ms injected latency (simulates VLM on edge)
    ros2 run lwsd_bench latency_injector \
        --ros-args \
        -p input_topic:=/odom \
        -p output_topic:=/odom_delayed \
        -p delay_ms:=500.0
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.timer import Timer

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from collections import deque
import time


class LatencyInjector(Node):
    """
    Delays a pose/odometry topic by a configurable amount.

    Stores incoming messages in a buffer and releases them
    after the configured delay. The original header timestamp
    is preserved so the LWSD monitor correctly computes
    inference latency.
    """

    def __init__(self):
        super().__init__("latency_injector")

        # Parameters
        self.declare_parameter("input_topic", "/odom")
        self.declare_parameter("output_topic", "/odom_delayed")
        self.declare_parameter("delay_ms", 200.0)
        self.declare_parameter("input_type", "odometry")  # 'odometry' or 'pose'

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value
        self.delay_s = self.get_parameter("delay_ms").value / 1000.0
        input_type = self.get_parameter("input_type").value

        # Buffer: (release_time, message)
        self.buffer: deque = deque()

        # QoS
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=50,
        )

        # Subscribe and publish
        if input_type == "odometry":
            self.sub = self.create_subscription(
                Odometry, input_topic, self._odom_cb, sensor_qos
            )
            self.pub = self.create_publisher(Odometry, output_topic, 10)
        else:
            self.sub = self.create_subscription(
                PoseStamped, input_topic, self._pose_cb, sensor_qos
            )
            self.pub = self.create_publisher(PoseStamped, output_topic, 10)

        # Timer to flush buffer at 100 Hz (fast enough for any control rate)
        self.flush_timer = self.create_timer(0.01, self._flush)

        self.msg_count = 0

        self.get_logger().info(
            f"Latency injector started.\n"
            f"  Input:  {input_topic}\n"
            f"  Output: {output_topic}\n"
            f"  Delay:  {self.delay_s*1000:.0f} ms"
        )

    def _odom_cb(self, msg: Odometry):
        """Buffer incoming odometry with release timestamp."""
        release_time = time.monotonic() + self.delay_s
        self.buffer.append((release_time, msg))

    def _pose_cb(self, msg: PoseStamped):
        """Buffer incoming pose with release timestamp."""
        release_time = time.monotonic() + self.delay_s
        self.buffer.append((release_time, msg))

    def _flush(self):
        """Release any messages whose delay has elapsed."""
        now = time.monotonic()
        while self.buffer and self.buffer[0][0] <= now:
            _, msg = self.buffer.popleft()
            # Publish with ORIGINAL timestamp preserved.
            # This is critical: the LWSD monitor uses the gap between
            # message timestamp and current time as inference latency.
            self.pub.publish(msg)
            self.msg_count += 1

            if self.msg_count % 100 == 0:
                self.get_logger().info(
                    f"Relayed {self.msg_count} msgs "
                    f"(buffer depth: {len(self.buffer)}, "
                    f"delay: {self.delay_s*1000:.0f}ms)"
                )


def main(args=None):
    rclpy.init(args=args)
    node = LatencyInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
