#!/bin/bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
# source ~/interbotix_ws/install/setup.bash

# ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=wxai

echo "Start Running"
# source ~/interbotix_ws/install/setup.bash
ros2 launch trossen_arm_moveit moveit.launch.py robot_model:=wxai ros2_control_hardware_type:=real

# python manipulator_server.py --widowx_ros2 --cam_ids 0