# PRD-06: ROS2 Integration

> Module: DEF-UAVDETR | Priority: P1
> Depends on: PRD-03
> Status: ⬜ Not started

## Objective
Wrap UAV-DETR as a ROS2 detector node for ANIMA robotics pipelines while keeping the model path paper-faithful and RGB-camera-first.

## Context
The paper is vision-only. This integration must preserve that assumption and expose the detector on ROS2 topics without over-extending into non-paper sensor fusion. ZED 2i camera support is in scope; Unitree L2 fusion is explicitly future work.

Paper references:
- §3.1: the model consumes RGB frames and emits detections
- §4.6: edge deployment overhead is a limitation, so the node should avoid unnecessary copies and offer frame skipping

## Acceptance Criteria
- [ ] ROS2 node subscribes to `sensor_msgs/Image` or `sensor_msgs/CompressedImage`.
- [ ] Node publishes detections as `vision_msgs/Detection2DArray` and optional debug overlays as `sensor_msgs/Image`.
- [ ] Launch file supports checkpoint path, confidence threshold, and frame-skip controls.
- [ ] Test: `uv run pytest tests/test_ros2_config.py -v` passes.

## Files to Create

| File | Purpose | Paper Ref | Est. Lines |
|------|---------|-----------|-----------|
| `src/anima_def_uavdetr/ros2/node.py` | Detector node | §3.1 | ~220 |
| `src/anima_def_uavdetr/ros2/messages.py` | Detection conversion helpers | — | ~100 |
| `src/anima_def_uavdetr/ros2/launch/uavdetr.launch.py` | Launch entrypoint | — | ~80 |
| `tests/test_ros2_config.py` | Config/launch tests | — | ~80 |

## Architecture Detail

### Inputs
- `/camera/image_raw: sensor_msgs/Image`
- `/camera/image_compressed: sensor_msgs/CompressedImage`

### Outputs
- `/def_uavdetr/detections: vision_msgs/Detection2DArray`
- `/def_uavdetr/debug_image: sensor_msgs/Image`

### Algorithm
```python
class DefUavDetrNode(Node):
    def __init__(self):
        super().__init__("def_uavdetr")
        self.predictor = build_predictor_from_params(self)
        self.subscription = self.create_subscription(Image, "/camera/image_raw", self.on_image, 10)
        self.publisher = self.create_publisher(Detection2DArray, "/def_uavdetr/detections", 10)

    def on_image(self, msg: Image) -> None:
        frame = ros_image_to_numpy(msg)
        detections = self.predictor.predict([frame])[0]
        self.publisher.publish(to_detection_array(msg.header, detections))
```

## Dependencies
```toml
rclpy = ">=3.0"
cv-bridge = "*"
vision-msgs = "*"
```

## Data Requirements
| Asset | Size | Path | Download |
|------|------|------|----------|
| Inference checkpoint | model-dependent | `/Volumes/AIFlowDev/RobotFlowLabs/models/def_uavdetr/checkpoints/uavdetr-best.pt` | Required |

## Test Plan
```bash
uv run pytest tests/test_ros2_config.py -v
```

## References
- Paper: §3.1 "Overall Architecture"
- Paper: §4.6 "Algorithm Failures and Limitations"
- Depends on: PRD-03
- Feeds into: PRD-07
