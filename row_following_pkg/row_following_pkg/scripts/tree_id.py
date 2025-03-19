##Cpmplete Navigation
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
from collections import deque
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.duration import Duration
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
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
        self.tree_counter = 1
        self.lidar_points = []
        self.create_subscription(LaserScan, '/scan_1', partial(self.lidar_callback, lidar_height=0.0), 10)
        self.create_subscription(LaserScan, '/scan_2', partial(self.lidar_callback, lidar_height=0.20), 10)
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)
        self.tree_marker_publisher = self.create_publisher(MarkerArray, 'tree_markers', 10)
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
        self.end_of_row = False
        self.turning = False
        self.in_grace_period = False
        self.grace_period_start_time = None
        self.is_second_turn=False
        # Turning parameters
        self.initial_yaw = None
        self.current_yaw = None
        self.turn_target_yaw = None
        self.tree_id_counter = 0
        self.detected_trees = []
        # Post-turn confirmation parameters
        self.post_turn_confirmation_duration = 10  # Duration in seconds to ignore "end of row" after a turn
        self.post_turn_start_time = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Pass count to track number of passes
        self.pass_count = 0
        self.exiting = False

        # Early detection threshold
        self.detection_distance_threshold = 0.15  # Adjust this as needed, lower values for closer detection
        self.tree_marker_publisher = self.create_publisher(MarkerArray, 'tree_markers', 10)
        # Row width tracking and forward movement state
        self.row_width_measurements = deque(maxlen=10)  # Store recent row width measurements
        self.row_width_forward_distance_traveled = 0.0  # Track distance traveled after 90-degree turn

        # Flags for controlling movement after 90-degree turn
        self.moving_forward_after_turn = False  # Flag to indicate forward movement after the first 90-degree turn
        self.second_90_degree_turn_initiated = False

        # Heading stabilization for end of row
        self.heading_window = deque(maxlen=10)  # Track last 10 heading angles for stabilization
        self.locked_heading = None  # Heading lock for end-of-row stabilization

    def transform_point_to_map(self, x, y):
        """ Transform (x, y) from the LiDAR/base frame to the map frame """
        point_in_base = PointStamped()
        point_in_base.header.frame_id = "base_link"  # or LiDAR frame if necessary
        point_in_base.header.stamp = self.get_clock().now().to_msg()
        point_in_base.point.x = x
        point_in_base.point.y = y
        point_in_base.point.z = 0.0

        try:
            # Transform point to the map frame
            return self.tf_buffer.transform(point_in_base, "map", timeout=rclpy.duration.Duration(seconds=1.0))
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException) as e:
            self.get_logger().warning(f"Transform error: {e}")
            return None

    def lidar_callback(self, scan_data, lidar_height):
        self.lidar_points = []
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

    def process_marker(self):
        # Example hard-coded point in 'base_link' frame
        point = [1.0, 2.0, 0.0]
        transformed_point = self.transform_point(point)
        if transformed_point:
            self.publish_marker(transformed_point)

    def transform_point(self, point):
        point_stamped = PointStamped()
        point_stamped.header.frame_id = "base_link"
        point_stamped.header.stamp = rclpy.time.Time()
        point_stamped.point.x = point[0]
        point_stamped.point.y = point[1]
        point_stamped.point.z = point[2]

        try:
            # Transform point to the 'map' frame
            transformed_point = self.tf_buffer.transform(point_stamped, "map")
            return transformed_point
        except TransformException as ex:
            self.get_logger().warn(f"Could not transform point: {ex}")
            return None
    
    def publish_tree_markers(self, points, labels):
        """ Publish markers for each tree cluster """
        marker_array = MarkerArray()
        unique_labels = set(labels)

        for label in unique_labels:
            if label == -1:
                continue  # Ignore noise

            # Extract points in the current cluster
            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]

            # Calculate cluster center (tree position)
            center_x = np.mean(cluster_points[:, 0])
            center_y = np.mean(cluster_points[:, 1])

            # Create and configure a marker for this tree
            marker = Marker()
            marker.header.frame_id = "map"  # Use "map" frame for testing
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "trees"
            marker.id = int(label)  # Use label as ID to make each marker unique
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = center_x
            marker.pose.position.y = center_y
            marker.pose.position.z = 0.0  # Assume trees are on a flat plane
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.5  # Radius of the marker
            marker.scale.y = 0.5
            marker.scale.z = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Log the marker position for debugging
            self.get_logger().info(f"Publishing marker for tree at ({center_x}, {center_y}) with ID {label}")

            # Add marker to the array
            marker_array.markers.append(marker)

        # Publish all tree markers
        self.tree_marker_publisher.publish(marker_array)
        self.get_logger().info(f"Published {len(marker_array.markers)} markers to RViz")

    def is_near_existing_tree(self, x, y, threshold=1.0):
        """ Check if a tree marker is already near the given point (x, y) """
        for tree_x, tree_y in self.detected_trees:
            if np.hypot(tree_x - x, tree_y - y) < threshold:
                return True
        return False

    def odom_callback(self, msg):
        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y ** 2 + orientation_q.z ** 2)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)
        self.heading_window.append(self.current_yaw)  # Store recent heading

    def run_dbscan(self):
        if len(self.lidar_points) > 0:
            all_points = np.vstack(self.lidar_points)
            dbscan = DBSCAN(eps=0.6, min_samples=13)
            labels = dbscan.fit_predict(all_points)
            self.calculate_lines(all_points, labels)
                        # Only publish markers if we are in row-following mode and pass_count is 0
            if self.pass_count == 0 and self.get_lookahead_point:
                self.publish_tree_markers(all_points, labels)

            # Clear points after processing
            #self.lidar_points = []
            #self.lidar_points = []  # Clear points after processing
    
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

            # Update row width with a running average
            row_width = abs(self.right_line_coeffs[1] - self.left_line_coeffs[1])
            if 6.0 <= row_width <= 10.0:
                self.row_width_measurements.append(row_width)
                self.get_logger().info(f"Row width detected: {row_width}. Calculating average...")
            else:
                self.get_logger().info(f"ignored row width {row_width}")
        else:
            self.no_trees_detected = True
    
    def control_loop(self):
        # if self.perform_second_90_degree_turn == True:
        #     self.run_dbscan
        #     lookahead_x, lookahead_y = self.get_lookahead_point()
        #     steering_angle = self.calculate_steering_angle(lookahead_x, lookahead_y)

        #     twist = Twist()
        #     twist.linear.x = self.linear_speed
        #     twist.angular.z = steering_angle
        #     self.get_logger().info(f"Moving towards lookahead point. Lookahead X: {lookahead_x}, Lookahead Y: {lookahead_y}, Steering Angle: {steering_angle}")
        #     self.vel_pub.publish(twist)  
        
        # Ignore "end of row" condition during post-turn confirmation period
        if self.post_turn_start_time and (time.time() - self.post_turn_start_time < self.post_turn_confirmation_duration):
            self.get_logger().info("In post-turn confirmation period; ignoring end of row.")
            self.run_dbscan()
            self.end_of_row = False
            self.exiting = False
            if not self.no_trees_detected:
                self.get_logger().info("Trees detected; exiting post-turn confirmation.")
                self.post_turn_start_time = None
                
            else:
                twist = Twist()
                twist.linear.x = self.linear_speed
                self.vel_pub.publish(twist)
            return

        if self.turning or (self.in_grace_period and time.time() - self.grace_period_start_time < 3):
            return
        else:
            self.in_grace_period = False
            
        self.run_dbscan()

        # Adjusted early detection for end of row
        #if self.no_trees_detected or self.check_end_of_row_early():

        if self.end_of_row == True and self.exiting == False:
            self.get_logger().info("exiting row")
            self.exit_row()

        if self.middle_line is None:
            self.get_logger().info("No path detected yet.")
            return

        if len(self.lidar_points) > 0:
            front_distances = []
            for points in self.lidar_points:
                angles = np.arctan2(points[:, 1], points[:, 0])
                front_mask = (points[:, 0] > 0) & (angles >= -np.deg2rad(85)) & (angles <= np.deg2rad(85))  # Consider points in front of the robot
                front_distances.extend(points[front_mask][:, 0])
                self.get_logger().info(f"front distance is: {len(front_distances)}")
                self.get_logger().info(f"angle: {len(angles)}")
            
            if len(front_distances) > 5:
                self.end_of_row = False
            else:
                twist = Twist()
                twist.angular.z = 0.0
                self.vel_pub.publish(twist)
                self.end_of_row = True
                # avg_front_distance = np.mean(front_distances)
                # self.get_logger().info(f"Average front distance: {avg_front_distance}")
                # if avg_front_distance > self.detection_distance_threshold * 2:  # Adjust the multiplier as needed
                #     self.get_logger().info("Front distance indicates end of row.")
                #     return True    
        else:
            self.end_of_row = True

        lookahead_x, lookahead_y = self.get_lookahead_point()
        steering_angle = self.calculate_steering_angle(lookahead_x, lookahead_y)
        self.check_end_of_row_early()
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = steering_angle
        #self.get_logger().info(f"Moving towards lookahead point. Lookahead X: {lookahead_x}, Lookahead Y: {lookahead_y}, Steering Angle: {steering_angle}")
        self.vel_pub.publish(twist)
        self.get_logger().info(f"pass count is: {self.pass_count}")

    def check_end_of_row_early(self):
        # Use stabilized heading for end-of-row detection
        stable_heading = np.mean(self.heading_window) if self.heading_window else self.current_yaw
        #self.get_logger().info(f"Using stabilized heading: {stable_heading}")
        # if self.middle_line:
        #     x_vals, y_vals = self.middle_line
        #     if len(x_vals) > 0 and np.min(x_vals) > self.detection_distance_threshold:
        #         # Lock the heading when end of row is detected
        #         self.locked_heading = stable_heading
        #         return True
        #self.get_logger().info(f"Using stabilized heading: {len(self.lidar_points)}")

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
        self.max_angular_speed = 0.3
        twist.angular.z = self.max_angular_speed
        self.vel_pub.publish(twist)
        self.timer = self.create_timer(0.1, self.perform_turn)

    def initiate_90_degree_turn(self):
        self.turning = True
        self.initial_yaw = self.current_yaw
        self.turn_target_yaw = (self.initial_yaw - math.pi / 2) % (2 * math.pi)  # Right turn
        twist = Twist()
        self.max_angular_speed = 0.3
        twist.angular.z = -self.max_angular_speed
        self.vel_pub.publish(twist)
        if self.timer: # timer reset for 90 degree turn
            self.timer.cancel()
            self.timr=None
        self.timer = self.create_timer(0.1, self.perform_90_degree_turn)

    def perform_turn(self):
        if self.angle_difference(self.current_yaw, self.initial_yaw) >= math.pi - 0.2:
            if self.timer: # timer reset for 90 degree turn
                self.timer.cancel()
                self.timr=None
            self.pass_count = 1
            self.stop_turn()

    def perform_90_degree_turn(self):
        self.get_logger().info(f"Performing 90-degree turn. Current yaw: {self.current_yaw}, Initial yaw: {self.initial_yaw}")
        angle_diff = self.angle_difference(self.current_yaw, self.initial_yaw)
        self.get_logger().info(f"Angle difference: {angle_diff}")
        yaw_diff = abs(self.current_yaw - self.turn_target_yaw)
        #if not self.is_second_turn and self.angle_difference(self.current_yaw, self.initial_yaw) >= (math.pi / 2) - 0.05:
        if not self.is_second_turn and (math.pi / 2) - 0.3 <= angle_diff <= (math.pi / 2) + 0.3 or yaw_diff <= 0.1:    
            if self.timer:
                self.timer.cancel()
                self.timer=None
            #self.pass_count = 0
            self.get_logger().info(f"performing 90 degree turn")
            self.stop_turn_and_move_forward()  # Start forward movement after completing the right turn
        else:
            self.get_logger().info("90-degree turn condition not met yet. Continuing to turn.")

    def stop_turn(self):
        twist = Twist()
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        # Increment pass count after completing a turn
        #self.pass_count = 1
        # Reset flags and start post-turn confirmation period
        if self.timer:
            self.timer.cancel()
            self.timer=None        
        self.turning = False
        self.no_trees_detected = False
        self.middle_line = None
        self.in_grace_period = True
        self.grace_period_start_time = time.time()
        self.post_turn_start_time = time.time()  # Start confirmation period
        self.get_logger().info(f"Turn completed. Starting post-turn confirmation period. Pass count: {self.pass_count}")

        # Cancel timer after turn completes
       # self.timer.cancel()

    def stop_turn_and_move_forward(self):
        twist = Twist()
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        self.row_width = 7.0  # Default row width if no measurements
        self.get_logger().info(f"Completed 90-degree turn. Moving forward by row width distance of: {self.row_width} meters")
        #if len(self.row_width_measurements) > 0:
         #   self.row_width = np.mean(self.row_width_measurements)
        #else:
        self.row_width_forward_distance_traveled = 0.0  # Reset forward distance traveled
        self.moving_forward_after_turn = True
        if self.timer:
            self.timer.cancel()
            self.timr=None
        self.timer = self.create_timer(0.1, self.move_forward_after_turn)

    def move_forward_after_turn(self):
        if self.second_90_degree_turn_initiated:
            return
        if self.moving_forward_after_turn:
            distance_step = self.linear_speed * 0.1  # Distance covered in one timer step
            self.row_width_forward_distance_traveled += distance_step
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0  # Use locked heading (no steering adjustment)
            self.vel_pub.publish(twist)
            #self.get_logger().info(f"row width is {self.row_width}, and distance moved is {self.row_width_forward_distance_traveled}")
            if self.row_width_forward_distance_traveled >= self.row_width:
                self.moving_forward_after_turn = False
                if self.timer:
                    self.timer.cancel()
                    self.timer=None
                self.stop_robot_and_initiate_second_90_degree_turn()
            
    def exit_row(self):
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = 0.0
        self.vel_pub.publish(twist)
        self.exit_dist = 4
        self.exiting = True
        self.forward_distance_traveled = 0.0  # Reset forward distance traveled
        self.moving_forward_after_row = True
        self.timer = self.create_timer(0.1, self.move_forward_after_row)

    def move_forward_after_row(self):
        if self.moving_forward_after_row:
            self.get_logger().info(f'clearing row by {self.exit_dist} meters, distance traveled{self.forward_distance_traveled}')
            distance_step = self.linear_speed * 0.1  # Distance covered in one timer step
            self.forward_distance_traveled += distance_step
            self.max_angular_speed = 0.0
            self.get_logger().info(f'linear velocity  {self.linear_speed},Steering Angle: {self.max_angular_speed}')
            twist = Twist()
            twist.linear.x = self.linear_speed
            twist.angular.z = 0.0
            if self.forward_distance_traveled >= self.exit_dist:
                self.moving_forward_after_row = False
                if self.timer:
                    self.timer.cancel()
                    self.timer=None
                if self.pass_count == 0:
                    #self.moving_forward_after_row = False
                    self.initiate_turn()
                    self.get_logger().info('starting 180 degree turn')
                    self.pass_count = 1
                else:
                    #self.moving_forward_after_row = False
                    self.initiate_90_degree_turn()
                    self.get_logger().info('starting 90 degree turn')
                    self.pass_count = 0
            
    def stop_robot_and_initiate_second_90_degree_turn(self):
        if self.second_90_degree_turn_initiated:
            return
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = -self.max_angular_speed
        self.vel_pub.publish(twist)
        self.get_logger().info("Reached row width distance after forward movement. Initiating second 90-degree turn.")
        self.moving_forward_after_turn = False
        # Initiate the second 90-degree right turn
        self.second_90_degree_turn_initiated = True
        self.initial_yaw = self.current_yaw
        self.turn_target_yaw = (self.initial_yaw - math.pi / 2) % (2 * math.pi)
        self.timer = self.create_timer(0.1, self.perform_second_90_degree_turn)

    def perform_second_90_degree_turn(self):
        if self.second_90_degree_turn_initiated and self.angle_difference(self.current_yaw, self.initial_yaw) >= (math.pi / 2) - 0.05:
            #self.stop_robot()
            self.get_logger().info("Second 90-degree turn completed. Robot stopped.")
            # Reset all turning-related flags
            self.turning = False
            self.moving_forward_after_turn = False
            self.second_90_degree_turn_initiated = False
            # Reset and prepare for row-following
            self.no_trees_detected = False
            self.middle_line = None
            self.post_turn_start_time = time.time()  # Start confirmation period to avoid immediate "end of row"
            self.row_width_forward_distance_traveled = 0.0
            # Cancel any timers to ensure clean control flow
           # self.timer.cancel()
            self.is_second_turn = True
            # Trigger the control loop to resume normal row-following
            self.get_logger().info("Resuming row-following after second 90-degree turn. Pass count reset to 0.")
            self.pass_count = 0
            if self.timer:
                self.timer.cancel()  # Cancel the timer after completing the task
                self.timer = None  # Reset the timer to None            
        #if self.second_90_degree_turn_initiated and self.angle_difference(self.current_yaw, self.initial_yaw) == (math.pi / 2) - 0.05:

            #self.in_grace_period=True
            self.stop_turn()  # Explicitly invoke control loop to resume normal operation

    def angle_difference(self, current, initial):
        diff = current - initial
        while diff < -math.pi:
            diff += 2 * math.pi
        while diff > math.pi:
            diff -= 2 * math.pi
        return abs(diff)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.linear_speed = 0.0
        self.max_angular_speed = 0.0
        self.vel_pub.publish(twist)
        self.get_logger().info("Second 90-degree turn completed. Robot stopped.")
        self.timer.cancel()  # Stop the timer to complete the operation

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
