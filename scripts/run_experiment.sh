#!/usr/bin/env bash
#
# run_experiment.sh
#
# Runs the LWSD latency injection experiment on ROSMASTER M1.
# Executes three trials: baseline, +200ms, +500ms.
#
# Prerequisites:
#   1. ROSMASTER M1 powered on and ROS2 running
#   2. lwsd_bench package built:  colcon build --packages-select lwsd_bench
#   3. Workspace sourced:         source install/setup.bash
#   4. Ground truth publisher running (see below)
#
# Usage:
#   chmod +x scripts/run_experiment.sh
#   ./scripts/run_experiment.sh
#
# What this does:
#   For each delay (0, 200, 500 ms):
#     1. Starts latency injector with configured delay
#     2. Starts LWSD monitor subscribing to delayed output
#     3. Records a ROS2 bag of all topics
#     4. Waits for you to run the robot along the test path
#     5. Press Ctrl+C to end the trial
#     6. Saves CSV + bag for that trial
#
# Ground truth setup:
#   In a separate terminal, run one of:
#
#   A) Manual waypoint mode (simplest):
#      ros2 run lwsd_bench ground_truth_publisher \
#        --ros-args -p mode:=waypoints -p waypoints_file:=waypoints.csv
#      Then press Enter each time the robot crosses a tape mark.
#
#   B) CSV replay mode (from video-extracted positions):
#      ros2 run lwsd_bench ground_truth_publisher \
#        --ros-args -p mode:=csv -p csv_file:=ground_truth.csv

set -euo pipefail

# --- Configuration ---
OUTPUT_DIR="${HOME}/lwsd_experiment_$(date +%Y%m%d_%H%M%S)"
DELAYS=(0 200 500)
ESTIMATE_TOPIC="/odom"
REFERENCE_TOPIC="/ground_truth/pose"
CONTROL_PERIOD="0.05"
ALERT_THRESHOLD="0.5"

mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  LWSD Latency Injection Experiment"
echo "=============================================="
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Delays to test:   ${DELAYS[*]} ms"
echo "  Estimate topic:   ${ESTIMATE_TOPIC}"
echo "  Reference topic:  ${REFERENCE_TOPIC}"
echo ""
echo "  Make sure the ground truth publisher is"
echo "  running in a separate terminal."
echo "=============================================="
echo ""

for DELAY in "${DELAYS[@]}"; do
    TRIAL_DIR="${OUTPUT_DIR}/delay_${DELAY}ms"
    mkdir -p "${TRIAL_DIR}"

    CSV_PATH="${TRIAL_DIR}/lwsd_log.csv"
    BAG_PATH="${TRIAL_DIR}/rosbag"

    echo "----------------------------------------------"
    echo "  Trial: ${DELAY} ms injected latency"
    echo "----------------------------------------------"
    echo ""
    echo "  1. Position the robot at the start mark."
    echo "  2. Press ENTER to begin this trial."
    echo "  3. Drive the robot along the test path."
    echo "  4. Press Ctrl+C to end the trial."
    echo ""
    read -p "  Press ENTER to start trial (${DELAY}ms delay)..."

    # Start bag recording in background
    echo "  Starting bag recording..."
    ros2 bag record -o "${BAG_PATH}" \
        "${ESTIMATE_TOPIC}" \
        "${REFERENCE_TOPIC}" \
        /odom_delayed \
        /lwsd/value \
        /lwsd/rate \
        /lwsd/latency_ms \
        /lwsd/state_error_mm \
        /lwsd/alert \
        /diagnostics \
        &
    BAG_PID=$!

    # Start latency injector
    echo "  Starting latency injector (${DELAY}ms)..."
    ros2 run lwsd_bench latency_injector \
        --ros-args \
        -p input_topic:="${ESTIMATE_TOPIC}" \
        -p output_topic:=/odom_delayed \
        -p delay_ms:="${DELAY}.0" \
        -p input_type:=odometry \
        &
    INJECTOR_PID=$!

    # Give injector a moment to start
    sleep 1

    # Start LWSD monitor
    echo "  Starting LWSD monitor..."
    echo "  CSV output: ${CSV_PATH}"
    echo ""
    ros2 run lwsd_bench lwsd_monitor \
        --ros-args \
        -p estimate_topic:=/odom_delayed \
        -p reference_topic:="${REFERENCE_TOPIC}" \
        -p control_period:="${CONTROL_PERIOD}" \
        -p alert_threshold:="${ALERT_THRESHOLD}" \
        -p input_type:=odometry \
        -p log_to_csv:=true \
        -p csv_path:="${CSV_PATH}"
    # This blocks until Ctrl+C

    # Clean up
    echo ""
    echo "  Stopping trial..."
    kill "${INJECTOR_PID}" 2>/dev/null || true
    kill "${BAG_PID}" 2>/dev/null || true
    wait "${INJECTOR_PID}" 2>/dev/null || true
    wait "${BAG_PID}" 2>/dev/null || true

    echo "  Trial ${DELAY}ms complete."
    echo "  CSV:  ${CSV_PATH}"
    echo "  Bag:  ${BAG_PATH}"
    echo ""

    # Brief pause between trials
    sleep 2
done

echo "=============================================="
echo "  Experiment complete!"
echo "  Results in: ${OUTPUT_DIR}"
echo ""
echo "  Files:"
for DELAY in "${DELAYS[@]}"; do
    echo "    ${OUTPUT_DIR}/delay_${DELAY}ms/lwsd_log.csv"
done
echo ""
echo "  Next step: analyze with plot_results.py"
echo "=============================================="
