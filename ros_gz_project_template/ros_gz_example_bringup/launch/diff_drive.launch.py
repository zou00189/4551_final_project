# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np

from ament_index_python.packages import get_package_share_directory
from launch.actions import ExecuteProcess
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    # Configure ROS nodes for launch

    # Setup project paths
    pkg_project_bringup = get_package_share_directory('ros_gz_example_bringup')
    pkg_project_gazebo = get_package_share_directory('ros_gz_example_gazebo')
    pkg_project_description = get_package_share_directory('ros_gz_example_description')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')


    # Load the SDF file from "description" package
    sdf_file  =  os.path.join(pkg_project_description, 'models', 'mecanum_drive', 'model.sdf')
    with open(sdf_file, 'r') as infp:
        robot_desc = infp.read()

    # Setup to launch the simulator and Gazebo world
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': PathJoinSubstitution([
            pkg_project_gazebo,
            'worlds',
            'diff_drive.sdf'
        ])}.items(),
    )


    # Set patrol logic for the long obstacle.
    patrol_node = Node(
        package='ros_gz_example_application',
        executable='obstacle_patrol.py',
        name='obstacle_patrol',
        output='screen',
        parameters=[
            {'use_sim_time': True}
        ]
    )

    # Bridge ROS topics and Gazebo messages for establishing communication
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        parameters=[{
            'config_file': os.path.join(pkg_project_bringup, 'config', 'ros_gz_example_bridge.yaml'),
            'qos_overrides./tf_static.publisher.durability': 'transient_local',
        }],
        output='screen'
    )

    object_sdf_path = os.path.join(
        pkg_project_description, 'models', 'box_object', 'model.sdf')

    # Set a reasonable region to initialize the object
    x_rand = np.random.uniform(low=-6.0, high=-8.0)
    y_rand = np.random.uniform(low=-2, high=2)
    z_height = 0.25 

    spawn_box_object = ExecuteProcess(
            cmd=['ros2', 'run', 'ros_gz_sim', 'create',
                '-name', 'movable_box_object',
                '-file', object_sdf_path,
                '-allow_renaming', 'true',
                '-x', str(x_rand),  # Pass coordinates as separate list items
                '-y', str(y_rand),
                '-z', str(z_height)
                ],
            output='screen'
        )

    return LaunchDescription([
        gz_sim,
        spawn_box_object,
        bridge,
        patrol_node
    ])
