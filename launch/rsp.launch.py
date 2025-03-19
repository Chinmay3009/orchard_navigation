from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Paths and package configurations
    package_name = 'orch_sim'
    map_file_path = 'maps/orch_new_save.yaml'  # Path to the saved map file
    
    # Set up paths to the necessary files
    pkg_share = FindPackageShare(package=package_name).find(package_name)
    default_urdf_model_path = os.path.join(pkg_share, 'urdf/nw.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/new.rviz')
    map_path = os.path.join(pkg_share, map_file_path)

    # Launch configuration variables
    urdf_model = LaunchConfiguration('urdf_model')
    rviz_config_file = LaunchConfiguration('rviz_config_file')
    slam_params_file = LaunchConfiguration('slam_params_file')
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')

    # Declare Launch Arguments
    declare_urdf_model_path_cmd = DeclareLaunchArgument(
        'urdf_model', default_value=default_urdf_model_path, description='Path to robot URDF file')
    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config_file', default_value=default_rviz_config_path, description='Path to RViz config file')
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file', default_value=os.path.join(pkg_share, 'config', 'mapper_params_online_async.yaml'),
        description='Path to SLAM parameters file')
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time', default_value='True', description='Use simulation time')

    # Map Server Node
    map_server_cmd = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time, 'yaml_filename': map_path}]
    )

    # AMCL Node for localization
    amcl_cmd = Node(
        package='nav2_amcl',
        executable='amcl',
        name='amcl',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        remappings=[('/scan', '/scan')]
    )

    # Move Base Node for path planning and navigation
   # move_base_cmd = Node(
    #    package='nav2_bringup',
     #   executable='bringup_launch.py',
       # output='screen',
      #  parameters=[{'use_sim_time': use_sim_time}],
    #)

    # Robot State Publisher Node
    robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_model])}]
    )

    # SLAM Toolbox Node
    slam_toolbox_cmd = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[slam_params_file],
        remappings=[('/scan', '/scan')]
    )

    # Launch Description
    ld = LaunchDescription()

    # Add Launch Arguments
    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_slam_params_file_cmd)
    ld.add_action(declare_use_sim_time_cmd)

    # Add Nodes to Launch Description
    ld.add_action(robot_state_publisher_cmd)
    ld.add_action(map_server_cmd)
    ld.add_action(amcl_cmd)
    #ld.add_action(move_base_cmd)
    ld.add_action(slam_toolbox_cmd)

    return ld

