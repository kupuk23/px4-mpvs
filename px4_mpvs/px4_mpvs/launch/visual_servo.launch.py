#!/usr/bin/env python
############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

__author__ = "Pedro Roque, Jaeyoung Lim"
__contact__ = "padr@kth.se, jalim@ethz.ch"

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument,OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition, UnlessCondition
from launch.condition import Condition
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import tempfile




def generate_launch_description():
    # Declare launch arguments
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='direct_allocation',
        description='Mode of the controller (rate, wrench, direct_allocation)'
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='',  # Default namespace is empty
        description='Namespace for all nodes'
    )

    setpoint_from_rviz_arg = DeclareLaunchArgument(
        'setpoint_from_rviz',
        default_value='true',
        description='Publish setpoint pose via rviz'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation time.",
    )

    mode = LaunchConfiguration('mode')
    namespace = LaunchConfiguration('namespace')
    setpoint_from_rviz = LaunchConfiguration('setpoint_from_rviz')

    return LaunchDescription([
        mode_arg,
        namespace_arg,
        setpoint_from_rviz_arg,
        use_sim_time_arg,
        Node(
            package='px4_mpvs',
            namespace=namespace,
            executable='mpvs_main', #mpvs_spacecraft
            name='mpvs_main', #mpvs_spacecraft
            output='screen',
            emulate_tty=True,
            parameters=[
                {'mode': mode},
                {'namespace': namespace},
                {'setpoint_from_rviz': setpoint_from_rviz},
                {'use_sim_time': LaunchConfiguration("use_sim_time")},
            ]
        ),
        # Node(package='px4_mpvs',
        #     namespace=namespace,
        #     executable='features_detector_node',
        #     name='features_detector_node',
        #     # output='screen',
        #     emulate_tty=True,
        #     parameters=[
        #         {'namespace': namespace},
        #         {'use_sim_time': LaunchConfiguration("use_sim_time")},
        #     ]
        # ),

        Node(
            package='px4_mpc',
            namespace=namespace,
            executable='rviz_pos_marker',
            name='rviz_pos_marker',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': namespace},
                {'use_sim_time': LaunchConfiguration("use_sim_time")},
            ],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        
        # Node(
        #     package='px4_mpvs',
        #     namespace=namespace,
        #     executable='test_pose_camera',
        #     name='test_pose_camera',
        #     output='screen',
        #     emulate_tty=True,
        #     parameters=[
        #         {'namespace': namespace},
        #         {'use_sim_time': LaunchConfiguration("use_sim_time")},
        #     ],
        #     condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        # ),
        Node(
            package='px4_mpvs',
            namespace=namespace,
            executable='test_setpoints',
            name='test_setpoints',
            output='screen',
            emulate_tty=True,
            parameters=[
                {'namespace': namespace},
                {'use_sim_time': LaunchConfiguration("use_sim_time")},
            ],
            condition=UnlessCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        
        Node(
            package='px4_offboard',
            namespace=namespace,
            executable='visualizer',
            name='visualizer',
            parameters=[
                {'namespace': namespace},
                {'use_sim_time': LaunchConfiguration("use_sim_time")},
            ],
            condition=IfCondition(LaunchConfiguration('setpoint_from_rviz'))
        ),
        
        # Simple RViz node without namespace patching (when namespace is empty)
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2'
        # ),
        
        # OpaqueFunction for namespace patching (when namespace is provided)
        # OpaqueFunction(
        #     function=launch_setup,
        #     condition=NamespaceNotEmptyCondition(LaunchConfiguration('namespace'))
        # ),
    ])

def patch_rviz_config(original_config_path, namespace):
    """
    Patch the RViz configuration file to replace the namespace placeholder with the actual namespace.
    """
    with open(original_config_path, 'r') as f:
        content = f.read()

    # Replace placeholder with actual namespace
    content = content.replace('__NS__', f'/{namespace}' if namespace else '')
    
    # Write to temporary file
    tmp_rviz_config = tempfile.NamedTemporaryFile(delete=False, suffix='.rviz')
    tmp_rviz_config.write(content.encode('utf-8'))
    tmp_rviz_config.close()

    return tmp_rviz_config.name

def launch_setup(context, *args, **kwargs):
    """
    Function to set up the launch context and patch the RViz configuration.
    """
    namespace = LaunchConfiguration('namespace').perform(context)
    rviz_config_path = os.path.join(get_package_share_directory('px4_mpvs'), 'config_new.rviz')
    patched_config = patch_rviz_config(rviz_config_path, namespace)

    return [
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', patched_config]
        )
    ]