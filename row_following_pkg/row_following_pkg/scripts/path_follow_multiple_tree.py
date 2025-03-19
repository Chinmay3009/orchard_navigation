###Path ffollower for multiple tree monitoring
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from math import atan2, sqrt, pi
import numpy as np

class MotionPlanner(Node):
    def __init__(self):
        super().__init__('motion_planner')

        # ROS2 Subscribers
        self.create_subscription(Path, '/astar_path', self.path_callback, 10)
        self.create_subscription(Odometry, '/wheel/odometry', self.odometry_callback, 10)
        self.create_subscription(LaserScan, '/scan3', self.lidar_callback, 10)  
        
        # ROS2 Publisher
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # State variables
        self.path = []
        self.current_position = None
        self.current_orientation = None
        self.current_waypoint_index = 0
        self.is_goal_reached = False
        self.obstacle_detected = False
        self.lookahead_distance = 0.5  # Lookahead distance for pure pursuit
        self.obstacle_threshold = 0.6  # Minimum safe distance from an obstacle

        self.get_logger().info("🚀 Motion Planner Initialized")

    def path_callback(self, msg):
        """Receives the planned path and stores it."""
        self.get_logger().info("📌 Received new path.")
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_waypoint_index = 0  # Reset to first waypoint
        self.is_goal_reached = False

    def lidar_callback(self, msg):
        """Detects obstacles in the robot's path."""
        min_distance = min(msg.ranges)
        self.obstacle_detected = min_distance < self.obstacle_threshold
        if self.obstacle_detected:
            self.get_logger().warn(f"⚠️ Obstacle detected at {min_distance:.2f}m! Slowing down.")

    def odometry_callback(self, msg):
        """Updates the robot's position and executes motion planning."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_position = (x, y)

        # Extract orientation (Yaw)
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y**2 + orientation_q.z**2)
        self.current_orientation = atan2(siny_cosp, cosy_cosp)

        # Execute motion planning
        self.follow_path()

    def follow_path(self):
        """Follow the A* path using a stabilized pure pursuit controller."""
        if not self.path or self.is_goal_reached:
            return

        if not self.current_position or not self.current_orientation:
            self.get_logger().warn("⏳ Waiting for odometry data...")
            return

        # ✅ Ensure the robot starts at the first waypoint in the path
        if self.current_waypoint_index == 0:
            self.current_waypoint_index = self.get_nearest_waypoint_index()

        # ✅ Get the next waypoint
        lookahead_point = self.get_lookahead_point()

        if lookahead_point is None:
            self.get_logger().warn("🚨 No valid lookahead point found. Stopping robot.")
            self.stop_robot()
            return

        # ✅ Compute movement commands
        linear_velocity, angular_velocity = self.compute_pure_pursuit(lookahead_point)

        # ✅ Prevent excessive angular velocity
        max_angular_velocity = 0.3  # Prevent oversteering
        angular_velocity = max(-max_angular_velocity, min(max_angular_velocity, angular_velocity))

        # ✅ If the robot reaches the last waypoint, stop moving
        if self.euclidean_distance(self.current_position, self.path[-1]) < 0.3:
            self.get_logger().info("🎯 Goal Reached!")
            self.stop_robot()
            self.is_goal_reached = True
            return

        self.publish_velocity(linear_velocity, angular_velocity)

    def get_nearest_waypoint_index(self):
        """Find the index of the waypoint closest to the robot's current position."""
        if not self.path or self.current_position is None:
            return 0  # Default to the first waypoint if no path exists

        min_distance = float('inf')
        nearest_index = 0

        for i, waypoint in enumerate(self.path):
            distance = sqrt(
                (waypoint[0] - self.current_position[0])**2 +
                (waypoint[1] - self.current_position[1])**2
            )
            if distance < min_distance:
                min_distance = distance
                nearest_index = i

        return nearest_index


    def get_lookahead_point(self):
        """Find the next waypoint ahead of the robot to prevent looping."""
        for i in range(self.current_waypoint_index, len(self.path)):
            waypoint = self.path[i]
            distance = self.euclidean_distance(self.current_position, waypoint)
            
            # ✅ Ensure the waypoint is in front of the robot
            angle_to_waypoint = atan2(waypoint[1] - self.current_position[1], waypoint[0] - self.current_position[0])
            angle_difference = self.normalize_angle(angle_to_waypoint - self.current_orientation)

            if distance >= self.lookahead_distance and abs(angle_difference) < pi / 2:
                self.current_waypoint_index = i  # ✅ Update the waypoint index
                return waypoint

        return self.path[-1]  # Return last point if none are found


    def compute_pure_pursuit(self, lookahead_point):
        """Compute smooth pursuit with reduced angular oscillations."""
        if lookahead_point is None:
            return 0.0, 0.0

        dx = lookahead_point[0] - self.current_position[0]
        dy = lookahead_point[1] - self.current_position[1]
        angle_to_lookahead = atan2(dy, dx)
        angle_error = self.normalize_angle(angle_to_lookahead - self.current_orientation)

        # ✅ Reduce oversteering by limiting angular gain
        K_angular = 1.2  # Adjusted to reduce oscillations
        linear_velocity = 0.5  # Constant forward speed
        angular_velocity = K_angular * angle_error

        return linear_velocity, angular_velocity



    def clear_past_waypoints(self):
        """Removes waypoints that the robot has already passed."""
        if not self.path:
            return

        while len(self.path) > 1 and self.euclidean_distance(self.current_position, self.path[0]) < 0.1:
            self.path.pop(0)  # Remove passed waypoints

    def euclidean_distance(self, p1, p2):
        """Computes Euclidean distance between two points."""
        return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def normalize_angle(self, angle):
        """Normalizes an angle to the range [-pi, pi]."""
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def publish_velocity(self, linear, angular):
        """Sends velocity commands to the robot."""
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.velocity_publisher.publish(twist)

    def stop_robot(self):
        """Stops the robot by setting velocity to zero."""
        self.publish_velocity(0.0, 0.0)
        self.get_logger().info("🛑 Robot Stopped.")


def main(args=None):
    rclpy.init(args=args)
    node = MotionPlanner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
