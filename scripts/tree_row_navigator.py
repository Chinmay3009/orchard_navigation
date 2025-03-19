#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
from sklearn.linear_model import LinearRegression

class TreeRowNavigator(Node):
    def __init__(self):
        super().__init__('tree_row_navigator')

        # Parameters for robot control
        self.kp = 0.5  # Proportional gain for angular velocity
        self.linear_speed = 0.2  # Constant forward speed

        # LiDAR angle range for left and right zones
        self.left_zone_min_angle = np.deg2rad(45)  # 45 degrees to the left
        self.right_zone_max_angle = np.deg2rad(-45)  # 45 degrees to the right

        # Create subscriber and publisher
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def laser_callback(self, msg):
        # Convert LaserScan data to 2D Cartesian coordinates
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])
            angle += msg.angle_increment

        points = np.array(points)

        if len(points) < 10:
            self.get_logger().warn('Not enough points to process.')
            return

        # Divide points into left and right zones
        left_points = []
        right_points = []
        angle = msg.angle_min

        for i, r in enumerate(msg.ranges):
            if r < msg.range_min or r > msg.range_max:
                continue

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            if angle >= self.left_zone_min_angle:
                left_points.append([x, y])
            elif angle <= self.right_zone_max_angle:
                right_points.append([x, y])

            angle += msg.angle_increment

        # Fit lines to left and right points (representing tree rows)
        left_line = self.fit_line(np.array(left_points))
        right_line = self.fit_line(np.array(right_points))

        if left_line is None or right_line is None:
            self.get_logger().warn('Could not fit lines to both rows.')
            return

        # Compute centerline and control error
        center_y = (left_line[1] + right_line[1]) / 2
        error = center_y  # Deviation from center

        # Control robot to follow the centerline
        angular_vel = -self.kp * error
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)

    def fit_line(self, points):
        if len(points) < 2:
            return None  # Not enough points to fit a line

        x = points[:, 0]
        y = points[:, 1]

        # Perform linear regression to find slope and intercept
        reg = LinearRegression().fit(x.reshape(-1, 1), y)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        return slope, intercept

def main(args=None):
    rclpy.init(args=args)
    navigator = TreeRowNavigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

