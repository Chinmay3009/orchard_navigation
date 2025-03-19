## WORKS WELL FOR ROW FOLLOWING

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from functools import partial

def fit_line(x, y): #takes x and y points, ignores outliers and fita a line
    ransac = RANSACRegressor()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    ransac.fit(x, y)
    line_coeff = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
    return line_coeff #returns slope and intercept

#Calculate the center line between the rows
def calculate_middle_line(left_coeffs, right_coeffs, x_range):
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    left_y_vals = left_coeffs[0] * x_vals + left_coeffs[1]
    right_y_vals = right_coeffs[0] * x_vals + right_coeffs[1]
    center_y_vals = (left_y_vals + right_y_vals) / 2
    return x_vals, center_y_vals

class PurePursuitNode(Node):
    def __init__(self):
        super().__init__('pure_pursuit_node')

        # Lidar data and parameters
        self.lidar_points = []

        self.create_subscription(LaserScan, '/scan_1', partial(self.lidar_callback, lidar_height=0.0), 10)
        self.create_subscription(LaserScan, '/scan_2', partial(self.lidar_callback, lidar_height=0.20), 10)
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)
    

        # Velocity control publisher
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.2, self.control_loop)

        #initialize Robot control parameters
        self.linear_speed = 0.2  # Fixed forward speed
        self.lookahead_distance = 1.0  # Fixed lookahead distance
        self.max_angular_speed = 0.3  # Max allowed angular speed
        self.middle_line = None
        self.left_line_coeffs = None
        self.right_line_coeffs = None
        self.initial_line_calculated = False
        self.no_trees_detected = False

#Process the incoming scan data from LazerScan
#Converts the scan_data.ranges (distance measurements) and corresponding angles from the LiDAR into arrays.
    def lidar_callback(self, scan_data, lidar_height):
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))
        
        #filter out individual data (outside lidar range)
        mask = (ranges > scan_data.range_min) & (ranges < scan_data.range_max)
        ranges = ranges[mask]
        angles = angles[mask]
        
        #convert lidar data to 2d cartesian coordinates
        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)
        z_points = np.full_like(x_points, lidar_height)

        points = np.column_stack((x_points, y_points, z_points))
        self.lidar_points.append(points)
   
    #clustering algorithm
    def run_dbscan(self):
        if len(self.lidar_points) > 0:
            all_points = np.vstack(self.lidar_points)
            dbscan = DBSCAN(eps=0.7, min_samples=15)
            labels = dbscan.fit_predict(all_points)

            self.calculate_lines(all_points, labels)
            self.lidar_points = []  # Clear points after processing
    
    #Iterates through unique clusters. Points in the left (positive y) and right (negative y) are separated into different lists.
    def calculate_lines(self, points, labels):
        unique_labels = set(labels)
        left_cluster_centers_x = []
        left_cluster_centers_y = []
        right_cluster_centers_x = []
        right_cluster_centers_y = []
        
        #average x and y values of clusters
        for label in unique_labels:
            if label == -1:
                continue

            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]
            x_points = cluster_points[:, 0]
            y_points = cluster_points[:, 1]

            avg_x = np.mean(x_points)
            avg_y = np.mean(y_points)

            # Only consider valid large clusters(10 points)
            if len(cluster_points) >= 10:
                if avg_y > 0:
                    left_cluster_centers_x.append(avg_x)
                    left_cluster_centers_y.append(avg_y)
                else:
                    right_cluster_centers_x.append(avg_x)
                    right_cluster_centers_y.append(avg_y)
        
        #if clusters of both sides are detected, it fits a line through it, otherwise no trees were detected
        if len(left_cluster_centers_x) > 1 and len(right_cluster_centers_x) > 1:
            self.left_line_coeffs = fit_line(left_cluster_centers_x, left_cluster_centers_y)
            self.right_line_coeffs = fit_line(right_cluster_centers_x, right_cluster_centers_y)

            x_range = [np.min(left_cluster_centers_x + right_cluster_centers_x), np.max(left_cluster_centers_x + right_cluster_centers_x)]
            self.middle_line = calculate_middle_line(self.left_line_coeffs, self.right_line_coeffs, x_range)
        else:
            self.no_trees_detected = True
    
    #Control
    def control_loop(self):
        self.run_dbscan()

        if self.no_trees_detected:
            self.stop_robot()
            self.get_logger().info("No trees detected, stopping robot.")
            return

        if self.middle_line is None:
            self.get_logger().info("No path detected yet.")
            return

        # Get the lookahead point along the middle line
        lookahead_x, lookahead_y = self.get_lookahead_point()

        # Calculate the angle and curvature required to move towards the lookahead point
        steering_angle = self.calculate_steering_angle(lookahead_x, lookahead_y)

        # Command the robot to move
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = steering_angle
        self.get_logger().info(f"Moving towards lookahead point. Lookahead X: {lookahead_x}, Lookahead Y: {lookahead_y}, Steering Angle: {steering_angle}")
        self.vel_pub.publish(twist)
    
    #look ahead point along the mid line
    def get_lookahead_point(self):
        # Extract the middle line points
        x_vals, y_vals = self.middle_line

        # Find the closest point ahead of the robot at the lookahead distance
        lookahead_idx = np.argmin(np.abs(x_vals - self.lookahead_distance))
        return x_vals[lookahead_idx], y_vals[lookahead_idx]

    def calculate_steering_angle(self, lookahead_x, lookahead_y):
        # Calculate the curvature and steering angle to the lookahead point
        distance_to_lookahead = np.hypot(lookahead_x, lookahead_y)

        # Calculate the required curvature for pure pursuit
        if distance_to_lookahead == 0:
            return 0.0

        curvature = 2 * lookahead_y / (distance_to_lookahead ** 2)
        steering_angle = np.clip(curvature * self.linear_speed, -self.max_angular_speed, self.max_angular_speed)

        return steering_angle

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        self.get_logger().info("Robot stopped.")

#Initialize ros2, creating the node and keeping it running
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
