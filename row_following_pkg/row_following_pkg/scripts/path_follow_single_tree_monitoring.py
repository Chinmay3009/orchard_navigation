import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from math import atan2, sqrt, pi
import time
import matplotlib.pyplot as plt
import json
import math

class PathFollower(Node):
    def __init__(self):
        super().__init__('path_follower')

        # ROS2 Subscribers and Publisher
        self.create_subscription(Path, '/astar_path', self.path_callback, 10)
        self.create_subscription(Odometry, '/wheel/odometry', self.odometry_callback, 10)
        self.create_subscription(LaserScan, '/scan3', self.lidar_callback, 10)  
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # State variables
        self.path = []
        self.current_position = None
        self.current_orientation = None
        self.current_waypoint_index = 0
        self.is_goal_reached = False
        self.goal_position = None
        self.obstacle_detected = False
        self.final_position = None  

        # Control parameters
        self.linear_speed = 0.5
        self.angular_speed_limit = 0.5
        self.reached_threshold = 0.01
        self.lookahead_distance = 0.2
        self.obstacle_distance_threshold = 0.5  

        # Manual Tree Information
        self.manual_tree_id = 73  # Manually set tree ID
        self.tree_goal_position = None  

        # Data Logging for Graphs
        self.cte_log = []
        self.time_log = []
        self.robot_path = []
        self.planned_path = []
        self.start_time = time.time()  

        # Load tree database
        self.tree_db_path = '/home/chinmay/second_ros2_ws/src/row_following_pkg/tree_db_2.json'
        self.tree_positions = self.load_tree_positions()

        # Fetch tree goal position immediately
        self.fetch_manual_tree_position()

    def load_tree_positions(self):
        """Load tree positions from the JSON database."""
        try:
            with open(self.tree_db_path, 'r') as f:
                tree_data = json.load(f)
            return {int(k): tuple(v['tree_pos']) for k, v in tree_data['trees'].items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load tree database: {e}")
            return {}

    def fetch_manual_tree_position(self):
        """Fetch the tree position from the JSON file using manually set tree ID."""
        if self.manual_tree_id in self.tree_positions:
            tree_x, tree_y = self.tree_positions[self.manual_tree_id]
            self.tree_goal_position = (tree_x, tree_y + 4.0) if tree_y < 0 else (tree_x, tree_y + 4.0) # Offset Y by 4m
            self.get_logger().info(f"Using Manual Tree ID: {self.manual_tree_id}, True Goal: {self.tree_goal_position}")
        else:
            self.get_logger().error(f"Manual Tree ID {self.manual_tree_id} not found in database!")
            self.tree_goal_position = None

    def path_callback(self, msg):
        """Store the planned path from A*."""
        self.get_logger().info("Received new path.")
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.planned_path = self.path.copy()  
        self.current_waypoint_index = self.get_nearest_waypoint_index()
        self.is_goal_reached = False

        if self.path:
            self.goal_position = self.path[-1]
            self.get_logger().info(f"Goal Position Set from Path: {self.goal_position}")

    def lidar_callback(self, msg):
        """Detect obstacles using LIDAR."""
        min_distance = min(msg.ranges)
        self.obstacle_detected = min_distance < self.obstacle_distance_threshold
        if self.obstacle_detected:
            self.get_logger().warn(f"Obstacle detected at distance: {min_distance:.2f}m!")

    def odometry_callback(self, msg):
        """Store the actual robot path and update movement."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_position = (x, y)
        self.robot_path.append(self.current_position)

        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y**2 + orientation_q.z**2)
        self.current_orientation = atan2(siny_cosp, cosy_cosp)

        self.follow_path()

    def get_nearest_waypoint_index(self):
        """Find the index of the waypoint closest to the robot."""
        if not self.path:
            return 0
        return min(range(len(self.path)), key=lambda i: sqrt(
            (self.path[i][0] - self.current_position[0])**2 + (self.path[i][1] - self.current_position[1])**2
        ))

    def get_lookahead_point(self):
        """Find the next waypoint at least 'lookahead_distance' away."""
        for i in range(self.current_waypoint_index, len(self.path)):
            waypoint = self.path[i]
            if sqrt((waypoint[0] - self.current_position[0])**2 +
                    (waypoint[1] - self.current_position[1])**2) >= self.lookahead_distance:
                return waypoint
        return self.path[-1]

    def calculate_cross_track_error(self):
        """Compute perpendicular distance from robot to nearest path segment."""
        if not self.path or len(self.path) < 2 or not self.current_position:
            return float('inf')

        nearest_index = self.get_nearest_waypoint_index()
        if nearest_index >= len(self.path) - 1:
            return 0.0  

        nearest_point = self.path[nearest_index]
        next_point = self.path[nearest_index + 1]

        x1, y1 = nearest_point
        x2, y2 = next_point
        x, y = self.current_position

        denominator = sqrt((y2 - y1)**2 + (x2 - x1)**2)
        if denominator == 0:
            return self.cte_log[-1] if self.cte_log else 0.0

        cte = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / denominator
        return cte

    def calculate_euclidean_distance_error(self):
        """Compute Euclidean distance error from goal."""
        if self.goal_position is None or self.current_position is None:
            return float('inf')
        return sqrt((self.goal_position[0] - self.current_position[0])**2 +
                    (self.goal_position[1] - self.current_position[1])**2)

    def calculate_final_error(self):
        """Compute Euclidean error between final robot position and tree location."""
        if self.final_position is None:
            self.get_logger().warn("Final robot position is missing! Using last recorded position.")
            self.final_position = self.current_position  # Use last known position

        if self.tree_goal_position is None:
            self.get_logger().warn("Tree goal position is missing! Using last known goal.")
            return

        # Ensure we compare against the adjusted goal
        goal_x, goal_y = self.tree_goal_position
        robot_x, robot_y = self.final_position

        # Compute the errors
        x_error = abs(goal_x - robot_x)
        y_error = abs(goal_y - robot_y)
        final_error = math.sqrt(x_error**2 + y_error**2)

        # Log correct values
        self.get_logger().info(f"Tree Goal (Adjusted): ({goal_x}, {goal_y})")
        self.get_logger().info(f"Robot Final Position: ({robot_x}, {robot_y})")
        self.get_logger().info(f"X Error: {x_error:.4f} meters, Y Error: {y_error:.4f} meters")
        self.get_logger().info(f"Final Euclidean Distance Error: {final_error:.4f} meters")




    def follow_path(self):
        if not self.path or self.is_goal_reached:
            return
        if not self.current_position or not self.current_orientation:
            self.get_logger().warn("Waiting for odometry data...")
            return

        self.current_waypoint_index = self.get_nearest_waypoint_index()
        lookahead_point = self.get_lookahead_point()

        if self.calculate_euclidean_distance_error() <= self.reached_threshold:
            if not self.is_goal_reached:
                self.get_logger().info("Goal Reached!")
                self.final_position = self.current_position if self.current_position else None
                self.stop_robot()
                self.calculate_final_error()
                self.is_goal_reached = True
            return

        dx = lookahead_point[0] - self.current_position[0]
        dy = lookahead_point[1] - self.current_position[1]
        angle_to_lookahead = atan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_lookahead - self.current_orientation)

        self.publish_velocity(self.linear_speed, 2.0 * angle_error)

    def normalize_angle(self, angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle   

    def publish_velocity(self, linear, angular):
        """Publish the linear and angular velocity to move the robot."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.velocity_publisher.publish(twist)

    def stop_robot(self):
        """Stop the robot by setting linear and angular velocity to zero."""
        self.publish_velocity(0.0, 0.0)
        self.get_logger().info("Robot Stopped.")
    


def main(args=None):
    rclpy.init(args=args)
    node = PathFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
