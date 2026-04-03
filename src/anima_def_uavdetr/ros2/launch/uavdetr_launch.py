"""ROS2 launch file for the DEF-UAVDETR detector node.

Usage (inside a ROS2 workspace):
    ros2 launch anima_def_uavdetr uavdetr.launch.py checkpoint:=/path/to/best.pt
"""

from __future__ import annotations


def generate_launch_description():
    """Build launch description — requires launch + launch_ros at import time."""
    try:
        from launch import LaunchDescription  # type: ignore[import-untyped]
        from launch.actions import DeclareLaunchArgument  # type: ignore[import-untyped]
        from launch_ros.actions import Node  # type: ignore[import-untyped]
    except ImportError:
        # Outside ROS2 workspace — return empty for testability
        return None

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "checkpoint", default_value="",
                description="Path to model checkpoint",
            ),
            DeclareLaunchArgument(
                "confidence", default_value="0.25",
                description="Detection confidence threshold",
            ),
            DeclareLaunchArgument(
                "device", default_value="auto",
                description="Compute device: auto|cuda|cpu",
            ),
            DeclareLaunchArgument(
                "frame_skip", default_value="0",
                description="Frames to skip between inferences",
            ),
            Node(
                package="anima_def_uavdetr",
                executable="def_uavdetr_node",
                name="def_uavdetr",
                output="screen",
                parameters=[
                    {
                        "checkpoint_path": "",
                        "confidence": 0.25,
                        "device": "auto",
                        "frame_skip": 0,
                    }
                ],
                remappings=[
                    ("/camera/image_raw", "/camera/image_raw"),
                    ("/def_uavdetr/detections", "/def_uavdetr/detections"),
                ],
            ),
        ]
    )
