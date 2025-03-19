import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Paths
    package_name = 'orch_sim'
    pkg_share = get_package_share_directory(package_name)
    world_path = os.path.join(pkg_share, 'worlds', 'orch.world')
    urdf_path = os.path.join(pkg_share, 'urdf', 'nw.urdf')
    rviz_config_path = os.path.join(pkg_share, 'rviz', 'new.rviz')
    
    # Declare Launch Arguments
    declare_world_cmd = DeclareLaunchArgument(
        name='world',
        default_value=world_path,
        description='Path to the world file'
    )
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='True',
        description='Use simulation time'
    )
    
    # Gazebo Simulation
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzserver.launch.py')
        ),
        launch_arguments={'world': world_path}.items()
    )
    
    start_gazebo_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gzclient.launch.py')
        )
    )
    
    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_path])}],
        output='screen'
    )
    
    # A* Path Planning Node
    astar_node = Node(
        package='your_package_name',  # Replace with your package
        executable='astar_path_planner',  # The Python file you wrote for A* algorithm
        name='astar_path_planner',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # Robot Control Node
    control_node = Node(
        package='row_following_pkg',  # Replace with your package
        executable='rrt.py',  # Python file for controlling robot along path
        name='robot_controller',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )
    
    # Create Launch Description
    ld = LaunchDescription()

    # Declare launch arguments
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_use_sim_time_cmd)
    
    # Add actions to launch
    ld.add_action(start_gazebo)
    ld.add_action(start_gazebo_client)
    ld.add_action(robot_state_publisher)
    ld.add_action(astar_node)
    ld.add_action(control_node)
    
    return ld

