"""
Launch file for LWSD experiment on ROSMASTER M1.

Runs three configurations sequentially or in parallel:
  1. Baseline (native latency)
  2. +200 ms injected
  3. +500 ms injected

Usage:
    # Full experiment with default ROSMASTER topics
    ros2 launch lwsd_bench experiment.launch.py

    # Custom topics
    ros2 launch lwsd_bench experiment.launch.py \
        estimate_topic:=/odom \
        reference_topic:=/ground_truth/pose

    # Single run with specific delay
    ros2 launch lwsd_bench experiment.launch.py delay_ms:=200.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # --- Arguments ---
            DeclareLaunchArgument(
                "estimate_topic",
                default_value="/odom",
                description="Robot state estimate topic (nav_msgs/Odometry)",
            ),
            DeclareLaunchArgument(
                "reference_topic",
                default_value="/ground_truth/pose",
                description="Ground truth reference topic (PoseStamped)",
            ),
            DeclareLaunchArgument(
                "delay_ms",
                default_value="0.0",
                description="Injected latency in ms (0 = native)",
            ),
            DeclareLaunchArgument(
                "control_period",
                default_value="0.05",
                description="Control period in seconds",
            ),
            DeclareLaunchArgument(
                "alert_threshold",
                default_value="0.5",
                description="LWSD alert threshold",
            ),
            DeclareLaunchArgument(
                "csv_path",
                default_value="/tmp/lwsd_log.csv",
                description="Path for LWSD CSV output",
            ),
            # --- Latency Injector ---
            # Only active when delay_ms > 0
            # When delay_ms = 0, monitor subscribes to raw estimate topic
            Node(
                package="lwsd_bench",
                executable="latency_injector",
                name="latency_injector",
                parameters=[
                    {
                        "input_topic": LaunchConfiguration("estimate_topic"),
                        "output_topic": "/odom_delayed",
                        "delay_ms": LaunchConfiguration("delay_ms"),
                        "input_type": "odometry",
                    }
                ],
                output="screen",
            ),
            # --- LWSD Monitor ---
            # Subscribes to delayed output when injector is active
            Node(
                package="lwsd_bench",
                executable="lwsd_monitor",
                name="lwsd_monitor",
                parameters=[
                    {
                        "estimate_topic": "/odom_delayed",
                        "reference_topic": LaunchConfiguration(
                            "reference_topic"
                        ),
                        "control_period": LaunchConfiguration(
                            "control_period"
                        ),
                        "alert_threshold": LaunchConfiguration(
                            "alert_threshold"
                        ),
                        "input_type": "odometry",
                        "log_to_csv": True,
                        "csv_path": LaunchConfiguration("csv_path"),
                    }
                ],
                output="screen",
            ),
        ]
    )
