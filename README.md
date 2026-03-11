# lwsd-bench

**Latency-Weighted State Divergence** diagnostic for foundation model robots.

Measures how quickly a robot's world model diverges from reality as a function of its perception pipeline latency.

Paper: *"Latency-Weighted State Divergence in Foundation Model Robots: Why Larger Models Do Not Guarantee Longer Safe Operation"*

## What This Does

`lwsd-bench` computes a single scalar (LWSD) that tells you how stale your robot's perception is, weighted by how fast the environment is changing relative to your control loop. It runs as a ROS2 node alongside your existing stack.

```
LWSD(t) = (state_error / control_period) * (inference_latency / control_period)
```

Higher LWSD = your robot is acting on a world model that does not match reality.

## Quick Start

### Install

```bash
cd ~/lwsd_bench_ws
colcon build --packages-select lwsd_bench
source install/setup.bash
```

### Run (online, live robot)

```bash
# Terminal 1: your robot stack (ROSMASTER M1 example)
ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py

# Terminal 2: ground truth publisher
ros2 run lwsd_bench ground_truth_publisher \
    --ros-args -p mode:=waypoints -p waypoints_file:=config/sample_waypoints.csv

# Terminal 3: LWSD monitor
ros2 run lwsd_bench lwsd_monitor \
    --ros-args \
    -p estimate_topic:=/odom \
    -p reference_topic:=/ground_truth/pose \
    -p control_period:=0.05
```

### Run (offline, from bag)

```bash
ros2 run lwsd_bench lwsd_bag_analyzer \
    --ros-args \
    -p bag_path:=/path/to/rosbag \
    -p estimate_topic:=/odom \
    -p reference_topic:=/ground_truth/pose \
    -p output_csv:=/tmp/lwsd_results.csv
```

## The Latency Injection Experiment

This is the experiment described in the paper. Same robot, same path, three latency conditions.

### Hardware needed

- Yahboom ROSMASTER M1 (or any ROS2 robot with odometry)
- Tape measure
- Masking tape (for floor marks)
- Camera on a tripod (for ground truth extraction, optional)

### Setup

1. Lay tape marks on the floor at 0.5 m intervals along a straight 4 m path.

2. Measure the exact position of each mark. Edit `config/sample_waypoints.csv`:
   ```
   x,y,z
   0.0,0.0,0.0
   0.5,0.0,0.0
   1.0,0.0,0.0
   ...
   ```

3. Place a movable object (box, mug) near the path. During each trial, slide it at roughly 0.3 m/s to create environment dynamics.

### Run the experiment

```bash
# Terminal 1: robot stack
ros2 launch yahboomcar_bringup yahboomcar_bringup_launch.py

# Terminal 2: ground truth publisher (press Enter at each tape mark)
ros2 run lwsd_bench ground_truth_publisher \
    --ros-args -p mode:=waypoints -p waypoints_file:=config/sample_waypoints.csv

# Terminal 3: run all three trials
chmod +x scripts/run_experiment.sh
./scripts/run_experiment.sh
```

The script runs three trials automatically:
- **Baseline (0 ms):** Native perception latency (~15-30 ms)
- **+200 ms:** Simulates VLM on cloud GPU (OpenVLA on A100)
- **+500 ms:** Simulates VLM on edge hardware (OpenVLA on Jetson)

### Analyze results

```bash
pip install matplotlib  # if not installed
python3 scripts/plot_results.py ~/lwsd_experiment_YYYYMMDD_HHMMSS
```

Generates:
- `lwsd_timeseries.pdf` - LWSD over time for all three conditions
- `lwsd_boxplot.pdf` - Distribution comparison
- `error_vs_latency.pdf` - State error vs latency with theoretical overlay
- `summary_table.tex` - LaTeX table ready to paste into the paper

## Nodes

### `lwsd_monitor`

Real-time LWSD computation. Subscribes to estimate and reference topics, publishes LWSD diagnostics.

**Published topics:**
| Topic | Type | Description |
|---|---|---|
| `/lwsd/value` | Float64 | Current LWSD value |
| `/lwsd/rate` | Float64 | LWSD time derivative |
| `/lwsd/latency_ms` | Float64 | Measured inference latency |
| `/lwsd/state_error_mm` | Float64 | Absolute state error |
| `/lwsd/alert` | String | Alert message (when threshold exceeded) |
| `/diagnostics` | DiagnosticArray | Standard ROS2 diagnostics |

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `estimate_topic` | `/odom` | State estimate topic |
| `reference_topic` | `/ground_truth/pose` | Reference / ground truth topic |
| `control_period` | `0.05` | Control cycle period (seconds) |
| `alert_threshold` | `0.5` | LWSD value that triggers alert |
| `input_type` | `odometry` | `odometry` or `pose` |
| `log_to_csv` | `true` | Write CSV log |
| `csv_path` | `/tmp/lwsd_log.csv` | CSV output path |

### `latency_injector`

Adds configurable delay to a pose/odometry topic. Preserves original timestamps so LWSD monitor correctly measures the injected latency.

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `input_topic` | `/odom` | Input topic |
| `output_topic` | `/odom_delayed` | Output (delayed) topic |
| `delay_ms` | `200.0` | Injected delay in milliseconds |

### `ground_truth_publisher`

Publishes ground truth poses. Two modes:
- **waypoints:** Manual trigger (press Enter at each tape mark)
- **csv:** Replay timestamped poses from a CSV file

### `lwsd_bag_analyzer`

Offline LWSD computation from recorded bag files.

## Package Structure

```
lwsd_bench/
  lwsd_bench/
    __init__.py
    core.py              # Pure LWSD computation (no ROS dependency)
    lwsd_monitor.py      # Online ROS2 monitor node
    latency_injector.py  # Adds artificial delay
    ground_truth_publisher.py  # Publishes reference poses
    bag_analyzer.py      # Offline bag analysis
  config/
    rosmaster_m1.yaml    # Default config for ROSMASTER M1
    sample_waypoints.csv # Example waypoint file
  launch/
    experiment.launch.py # Launch file for the experiment
  scripts/
    run_experiment.sh    # Automated experiment runner
    plot_results.py      # Generate figures from CSV data
  test/
    test_core.py         # Unit tests for LWSD computation
```

## Citation

```bibtex
@article{dholakia2025lwsd,
  title={Latency-Weighted State Divergence in Foundation Model Robots:
         Why Larger Models Do Not Guarantee Longer Safe Operation},
  author={Dholakia, Stavan and Singh, Abhishek and Gazta, Aditya and Shukla, Shivani},
  journal={arXiv preprint},
  year={2025}
}
```

## License

MIT
