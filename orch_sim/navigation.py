#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import math as m
import numpy as np

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class InRowNavigation(Node):
    def __init__(self):
        super().__init__('inrow_navigation')

        # Initialize properties for in-row navigation
        self.laserscanner_data = None
        self.new_scan_available = False  # Tracks if new laser scan data is available

        # Create a publisher for velocity commands to control the robot
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscribe to laser scan data
        self.create_subscription(LaserScan, '/scan', self.laserscan_callback, 10)

        # Create a timer to run the main loop at 10 Hz
        self.create_timer(0.1, self.main_loop)  # 0.1 sec = 10 Hz

    def main_loop(self):
        if self.new_scan_available:
            # If new laser scan data is available, process it
            self.new_scan_available = False
            self.process_laserscan()

    def laserscan_callback(self, scan_data):
        """Callback to store laser scan data when received."""
        self.laserscanner_data = scan_data
        self.new_scan_available = True

    def process_laserscan(self):
        """Process laser scan data to calculate the robot's lateral and angular deviations."""
        ranges = np.array(self.laserscanner_data.ranges)
        angle_min = self.laserscanner_data.angle_min
        angle_increment = self.laserscanner_data.angle_increment

        # Determine left and right plant points based on laser scan data
        ranges_left = ranges[30:170]  # Left side (adjust indices as necessary)
        ranges_right = ranges[-170:-30]  # Right side (adjust indices as necessary)
        ranges_right = np.flip(ranges_right)  # Flip the right-side data for consistency

        # Convert laser ranges to x, y coordinates
        points_left = self.extract_coordinates(ranges_left, angle_min + 30 * angle_increment, angle_increment)
        points_right = self.extract_coordinates(ranges_right, angle_min + (len(ranges) - 170) * angle_increment, angle_increment)

        # Fit a line to the left and right points using least-squares
        m_left, c_left = self.fit_line(points_left)
        m_right, c_right = self.fit_line(points_right)

        # Calculate the angular and lateral deviations
        if m_left is not None and m_right is not None:
            self.control_robot(m_left, c_left, m_right, c_right)

    def extract_coordinates(self, ranges, angle_min, angle_increment):
        """Extract x, y coordinates from ranges in laser scan data."""
        x_points = []
        y_points = []

        for i, r in enumerate(ranges):
            if r > 0.05 and r < 10.0:  # Filter out invalid or very distant points
                angle = angle_min + i * angle_increment
                x_points.append(r * m.cos(angle))
                y_points.append(r * m.sin(angle))

        return np.array(x_points), np.array(y_points)

    def fit_line(self, points):
        """Fit a line to the given x, y points using least squares."""
        x_points, y_points = points
        if len(x_points) > 10:  # Ensure we have enough points
            A = np.vstack([x_points, np.ones(len(x_points))]).T
            m, c = np.linalg.lstsq(A, y_points, rcond=None)[0]  # Solve y = mx + c
            return m, c
        else:
            return None, None

    def control_robot(self, m_left, c_left, m_right, c_right):
        """Calculate and publish velocity commands to stay centered between the rows."""
        # Calculate the centerline's deviation based on the left and right line intercepts
        centerline_y = (c_left + c_right) / 2.0

        # Calculate the angular deviation
        angular_deviation = (m.atan(m_right) - m.atan(m_left)) / 2.0

        # Proportional control constants (you can adjust these values)
        k_linear = 0.5
        k_angular = 2.0

        # Calculate the velocity commands
        linear_velocity = k_linear * 0.5  # Constant forward velocity
        angular_velocity = -k_angular * angular_deviation

        # Limit angular velocity to prevent sharp turns
        max_angular_velocity = 1.0
        angular_velocity = max(min(angular_velocity, max_angular_velocity), -max_angular_velocity)

        # Publish the velocity commands
        cmd = Twist()
        cmd.linear.x = linear_velocity
        cmd.angular.z = angular_velocity
        self.cmd_pub.publish(cmd)

        self.get_logger().info(f"Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")
        self.get_logger().info(f"Centerline deviation: {centerline_y}, Angular Deviation: {angular_deviation}")

def main(args=None):
    rclpy.init(args=args)
    node = InRowNavigation()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()

