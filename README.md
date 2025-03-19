# Orchard Simulation and Autonomous Navigation in ROS2

This repository contains a Gazebo simulation for an orchard environment and a ROS2-based navigation package for autonomous robot mapping and tree monitoring.

## Project Structure

- **orch_sim** – Gazebo simulation of the orchard along with the robot.  
- **row_following_pkg** – Scripts for navigation, mapping, and A* path planning.  

## Installation & Setup

### Launching the Simulation
To start the simulation in Linux, run the following command:
- `ros2 launch orch_sim/launch/launch_urdf_into_gazebo.launch`

The starting position of the robot can be modified in the launch file (`launch_urdf_into_gazebo.launch`).

## Usage

### Mapping the Orchard
1. Navigate to `row_following_pkg/row_following_pkg/scripts`.  
2. Open `orch_sim/config/mapper_params_online_async.yaml` and set `mode: mapping`.  
3. Run `tree_marking.py`.  
4. Run `autonomous_navigation.py`.  

### Single Tree Monitoring
1. Launch the simulation.  
2. Open `orch_sim/config/mapper_params_online_async.yaml` and set `mode: localization`.  
3. Run `path_follow_single_tree_monitoring.py`.  
4. Run `single_tree_monitoring.py`.  

### Multiple Tree Monitoring
1. Launch the simulation.  
2. Open `orch_sim/config/mapper_params_online_async.yaml` and set `mode: localization`.  
3. Run `path_follow_multiple_tree.py`.  
4. Run `multiple_tree_monitoring.py`.  

## Notes
- Ensure ROS2 and Gazebo are properly installed before running the scripts.  
- The robot's starting position can be modified in `launch_urdf_into_gazebo.launch`.  
- Modify the mapping/localization modes in `mapper_params_online_async.yaml` accordingly.  

## Contributing
Contributions, suggestions, and issue reports are welcome. Feel free to submit pull requests or raise issues for improvements.

## License
This project is licensed under the MIT License.

