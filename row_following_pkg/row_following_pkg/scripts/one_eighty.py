## WORKS WELL FOR 180 DEGREE TURNS AND ROW FOLLOWING##

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from functools import partial
import math
import time

def fit_line(x, y):
    ransac = RANSACRegressor()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    ransac.fit(x, y)
    line_coeff = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
    return line_coeff

def calculate_middle_line(left_coeffs, right_coeffs, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    left_y_vals = left_coeffs[0] * x_vals + left_coeffs[1]
    right_y_vals = right_coeffs[0] * x_vals + right_coeffs[1]
    center_y_vals = (left_y_vals + right_y_vals) / 2
    return x_vals, center_y_vals

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')
        
        # Initialize LiDAR data
        self.lidar_points = []
        self.create_subscription(LaserScan, '/scan_1', partial(self.lidar_callback, lidar_height=0.0), 10)
        self.create_subscription(LaserScan, '/scan_2', partial(self.lidar_callback, lidar_height=0.20), 10)
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)
        
        # Odometry subscriber for tracking rotation
        self.create_subscription(Odometry, '/wheel/odometry', self.odom_callback, 10)

        # Velocity publisher
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.2, self.control_loop)

        # Control parameters
        self.linear_speed = 0.4
        self.lookahead_distance = 2.0
        self.max_angular_speed = 0.3
        self.middle_line = None
        self.no_trees_detected = False
        self.turning = False
        self.in_grace_period = False
        self.grace_period_start_time = None

        # Turning parameters
        self.initial_yaw = None
        self.current_yaw = None
        self.turn_target_yaw = None

        # Post-turn confirmation parameters
        self.post_turn_confirmation_duration = 3  # Duration in seconds to ignore "end of row" after a turn
        self.post_turn_start_time = None

    def lidar_callback(self, scan_data, lidar_height):
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
        mask = (ranges > scan_data.range_min) & (ranges < scan_data.range_max)
        ranges = ranges[mask]
        angles = angles[mask]
        
        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)
        z_points = np.full_like(x_points, lidar_height)
        points = np.column_stack((x_points, y_points, z_points))
        self.lidar_points.append(points)
    
    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y ** 2 + orientation_q.z ** 2)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def run_dbscan(self):
        if len(self.lidar_points) > 0:
            all_points = np.vstack(self.lidar_points)
            dbscan = DBSCAN(eps=0.7, min_samples=10)
            labels = dbscan.fit_predict(all_points)
            self.calculate_lines(all_points, labels)
            self.lidar_points = []  # Clear points after processing
    
    def calculate_lines(self, points, labels):
        unique_labels = set(labels)
        left_cluster_centers_x = []
        left_cluster_centers_y = []
        right_cluster_centers_x = []
        right_cluster_centers_y = []
        
        for label in unique_labels:
            if label == -1:
                continue

            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]
            x_points = cluster_points[:, 0]
            y_points = cluster_points[:, 1]

            avg_x = np.mean(x_points)
            avg_y = np.mean(y_points)

            if len(cluster_points) >= 10:
                if avg_y > 0:
                    left_cluster_centers_x.append(avg_x)
                    left_cluster_centers_y.append(avg_y)
                else:
                    right_cluster_centers_x.append(avg_x)
                    right_cluster_centers_y.append(avg_y)
        
        if len(left_cluster_centers_x) > 1 and len(right_cluster_centers_x) > 1:
            self.left_line_coeffs = fit_line(left_cluster_centers_x, left_cluster_centers_y)
            self.right_line_coeffs = fit_line(right_cluster_centers_x, right_cluster_centers_y)
            x_range = [np.min(left_cluster_centers_x + right_cluster_centers_x), np.max(left_cluster_centers_x + right_cluster_centers_x)]
            self.middle_line = calculate_middle_line(self.left_line_coeffs, self.right_line_coeffs, x_range)
            self.no_trees_detected = False
        else:
            self.no_trees_detected = True
    
    def control_loop(self):
        # Ignore "end of row" condition during post-turn confirmation period
        if self.post_turn_start_time and (time.time() - self.post_turn_start_time < self.post_turn_confirmation_duration):
            self.get_logger().info("In post-turn confirmation period; ignoring end of row.")
            self.run_dbscan()
            if not self.no_trees_detected:
                # Trees detected; reset end-of-row state and exit confirmation period
                self.get_logger().info("Trees detected; exiting post-turn confirmation.")
                self.post_turn_start_time = None
            else:
                # Continue moving forward during confirmation period
                twist = Twist()
                twist.linear.x = self.linear_speed
                self.vel_pub.publish(twist)
            return

        if self.turning or (self.in_grace_period and time.time() - self.grace_period_start_time < 3):
            return

        self.run_dbscan()

        if self.no_trees_detected:
            self.initiate_turn()
            return

        if self.middle_line is None:
            self.get_logger().info("No path detected yet.")
            return

        lookahead_x, lookahead_y = self.get_lookahead_point()
        steering_angle = self.calculate_steering_angle(lookahead_x, lookahead_y)

        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = steering_angle
        self.get_logger().info(f"Moving towards lookahead point. Lookahead X: {lookahead_x}, Lookahead Y: {lookahead_y}, Steering Angle: {steering_angle}")
        self.vel_pub.publish(twist)
    
    def get_lookahead_point(self):
        x_vals, y_vals = self.middle_line
        lookahead_idx = np.argmin(np.abs(x_vals - self.lookahead_distance))
        return x_vals[lookahead_idx], y_vals[lookahead_idx]

    def calculate_steering_angle(self, lookahead_x, lookahead_y):
        distance_to_lookahead = np.hypot(lookahead_x, lookahead_y)
        if distance_to_lookahead == 0:
            return 0.0

        curvature = 2 * lookahead_y / (distance_to_lookahead ** 2)
        steering_angle = np.clip(curvature * self.linear_speed, -self.max_angular_speed, self.max_angular_speed)
        return steering_angle

    def initiate_turn(self):
        self.turning = True
        self.initial_yaw = self.current_yaw
        self.turn_target_yaw = (self.initial_yaw + math.pi) % (2 * math.pi)
        self.get_logger().info("Initiating 180-degree turn due to no trees detected.")

        twist = Twist()
        twist.angular.z = self.max_angular_speed
        self.vel_pub.publish(twist)
        self.timer = self.create_timer(0.1, self.perform_turn)

    def perform_turn(self):
        if self.angle_difference(self.current_yaw, self.initial_yaw) >= math.pi - 0.05:
            self.stop_turn()

    def angle_difference(self, current, initial):
        diff = current - initial
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        return abs(diff)

    def stop_turn(self):
        twist = Twist()
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)

        # Reset flags and start post-turn confirmation period
        self.turning = False
        self.no_trees_detected = False
        self.middle_line = None
        self.in_grace_period = True
        self.grace_period_start_time = time.time()
        self.post_turn_start_time = time.time()  # Start confirmation period
        self.get_logger().info("Turn completed. Starting post-turn confirmation period.")

        self.timer.cancel()

def main(args=None):
    rclpy.init(args=args)
    node = PurePursuitNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
