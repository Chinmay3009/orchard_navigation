import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Constants for paths to different files and folders
    gazebo_models_path = 'models'
    package_name = 'orch_sim'
    robot_name_in_model = 'two_wheeled_robot'
    rviz_config_file_path = 'rviz/new.rviz'
    urdf_file_path = 'urdf/nw.urdf'
    world_file_path = 'worlds/empty.world'
    
    # Pose where we want to spawn the robot
    spawn_x_val = '-2.0'
    spawn_y_val = '4.0'
    spawn_z_val = '0.0'
    spawn_yaw_val = '0.0'
    
    # Set the path to different files and folders  
    pkg_gazebo_ros = FindPackageShare(package='gazebo_ros').find('gazebo_ros')   
    pkg_share = FindPackageShare(package=package_name).find(package_name)
    default_urdf_model_path = os.path.join(pkg_share, urdf_file_path)
    default_rviz_config_path = os.path.join(pkg_share, rviz_config_file_path)  # Full path to the RViz config file
    world_path = os.path.join(pkg_share, world_file_path)
    gazebo_models_path = os.path.join(pkg_share, gazebo_models_path)
    os.environ["GAZEBO_MODEL_PATH"] = gazebo_models_path
    
    # Launch configuration variables specific to simulation
    gui = LaunchConfiguration('gui')
    headless = LaunchConfiguration('headless')  # Declare headless here properly
    rviz_config_file = LaunchConfiguration('rviz_config_file')  # Declare RViz config file
    urdf_model = LaunchConfiguration('urdf_model')
    use_sim_time = LaunchConfiguration('use_sim_time', default='True')

    # Declare the slam_params_file argument
    declare_slam_params_file_cmd = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(get_package_share_directory("orch_sim"),
                                   'config', 'mapper_params_online_async.yaml'),
        description='Full path to the ROS2 parameters file to use for the slam_toolbox node')

    # Declare Launch arguments
    declare_use_joint_state_publisher_cmd = DeclareLaunchArgument(
        name='gui',
        default_value='True',
        description='Flag to enable joint_state_publisher_gui')

    declare_headless_cmd = DeclareLaunchArgument(
        name='headless',
        default_value='False',
        description='Whether to execute gzclient in headless mode (no GUI)')

    declare_urdf_model_path_cmd = DeclareLaunchArgument(
        name='urdf_model', 
        default_value=default_urdf_model_path, 
        description='Absolute path to robot urdf file')
        
    declare_rviz_config_file_cmd = DeclareLaunchArgument(  # Declare rviz_config_file here
        name='rviz_config_file',
        default_value=default_rviz_config_path,
        description='Full path to the RViz config file to use')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        name='use_rviz',
        default_value='True',
        description='Whether to start RVIZ')

    declare_use_sim_time_cmd = DeclareLaunchArgument(
        name='use_sim_time',
        default_value='True',
        description='Use simulation (Gazebo) clock if true')

    declare_world_cmd = DeclareLaunchArgument(
        name='world',
        default_value=world_path,
        description='Full path to the world model file to load')
    
    # Start robot state publisher
    start_robot_state_publisher_cmd = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_model])}])

    # Start joint state publisher
    start_joint_state_publisher_cmd = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        condition=UnlessCondition(gui))

    # Start RViz
    start_rviz_cmd = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config_file])  # Reference the rviz_config_file here

    # Start Gazebo server
    start_gazebo_server_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzserver.launch.py')),
        launch_arguments={'world': world_path}.items())

    # Start Gazebo client    
    start_gazebo_client_cmd = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_gazebo_ros, 'launch', 'gzclient.launch.py')),
        condition=IfCondition(PythonExpression([headless, ' == False'])))

 #   delete_entity_cmd = Node(
  #      package='gazebo_ros',
   #     executable='delete_entity',
    #    arguments=['-entity', 'two_wheeled_robot'],
     #   output='screen'
    #)    
    
    # Launch the robot
    spawn_entity_cmd = Node(
        package='gazebo_ros', 
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'two_wheeled_robot', 
            '-topic', 'robot_description',
            '-x', spawn_x_val,
            '-y', spawn_y_val,
            '-z', spawn_z_val,
            '-Y', spawn_yaw_val],
        output='screen'
    )

    # SLAM Toolbox node with slam_params_file
    start_slam_toolbox_cmd = Node(
        package='slam_toolbox',
        executable='sync_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[LaunchConfiguration('slam_params_file')],  # Use the parameter file from launch argument
        remappings=[('/scan', '/scan')]
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_joint_state_publisher_cmd)
    ld.add_action(declare_headless_cmd)
    ld.add_action(declare_urdf_model_path_cmd)
    ld.add_action(declare_rviz_config_file_cmd)  
    ld.add_action(declare_use_rviz_cmd) 
    ld.add_action(declare_use_sim_time_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_slam_params_file_cmd)  

    # Add any actions
    ld.add_action(start_gazebo_server_cmd)
    ld.add_action(start_gazebo_client_cmd)
    ld.add_action(spawn_entity_cmd)
    ld.add_action(start_robot_state_publisher_cmd)
    ld.add_action(start_joint_state_publisher_cmd)
    ld.add_action(start_rviz_cmd)

    # Add SLAM actions
    ld.add_action(start_slam_toolbox_cmd)

    return ld 
