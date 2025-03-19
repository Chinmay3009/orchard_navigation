import numpy as np
from sklearn.linear_model import LinearRegression
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class RowFollowingNode(Node):
    def __init__(self):
        super().__init__('row_following_node')
        
        # Publisher to send velocity commands
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriber to receive LiDAR data
        self.laser_subscriber = self.create_subscription(
            LaserScan, '/scan', self.laser_callback, 10)
        
        # Parameters
        self.initial_trees_detected = False  # Flag for initial detection
        self.global_left_line = None  # Left row line equation (slope, intercept)
        self.global_right_line = None  # Right row line equation (slope, intercept)
        self.epsilon_distance = 0.5  # Distance between LiDAR points for DBSCAN
        self.min_points_for_line = 5  # Minimum number of points to fit a line
        self.empty_threshold = 10  # Threshold to consider large empty spaces

    def laser_callback(self, msg):
        # Convert LaserScan data into Cartesian coordinates (x, y)
        points = []
        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                points.append([x, y])
            angle += msg.angle_increment
        
        points = np.array(points)
        
        if len(points) < self.min_points_for_line:
            self.get_logger().warn('Not enough points to process.')
            return
        
        # Split points into left and right sets based on y-coordinate
        left_points, right_points = self.split_left_right(points)

        # Initially detect tree rows and fit global lines
        if not self.initial_trees_detected:
            if len(left_points) >= self.min_points_for_line and len(right_points) >= self.min_points_for_line:
                # Fit global lines based on initial points
                self.global_left_line = self.fit_line(left_points)
                self.global_right_line = self.fit_line(right_points)
                self.initial_trees_detected = True
                self.get_logger().info('Initial tree rows detected and lines fitted.')
        else:
            # Continue moving along the fitted global lines
            self.navigate_between_lines(self.global_left_line, self.global_right_line)
        
        # Check for large empty spaces to handle row end detection
        self.check_for_empty_space(points)

    def split_left_right(self, points):
        """Split the points into left and right sets based on y-coordinate."""
        left_points = []
        right_points = []
        for x, y in points:
            if y > 0:
                right_points.append([x, y])
            else:
                left_points.append([x, y])
        return np.array(left_points), np.array(right_points)

    def fit_line(self, points):
        """Fit a straight line (y = mx + c) using linear regression."""
        if len(points) < self.min_points_for_line:
            return None
        x = points[:, 0]
        y = points[:, 1]
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return m, c

    def navigate_between_lines(self, left_line, right_line):
        """Navigate the robot by calculating the center between left and right lines."""
        if left_line is None or right_line is None:
            return  # If either line is not detected, do nothing

        # Calculate the y-intercepts (at x = 0) for both lines
        left_slope, left_intercept = left_line
        right_slope, right_intercept = right_line

        # Compute the midpoint (centerline) between the two lines
        center_y = (left_intercept + right_intercept) / 2
        error = center_y  # Deviation from the center

        # Apply proportional control to adjust the angular velocity
        angular_vel = -0.5 * error  # Tune the gain for smoother control

        # Publish the velocity command
        twist = Twist()
        twist.linear.x = 0.3  # Constant forward speed
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)

    def check_for_empty_space(self, points):
        """Check for large empty spaces in the point cloud, indicating a gap or row end."""
        # Count how many points are within a reasonable distance
        close_points = np.sum(np.linalg.norm(points, axis=1) < self.epsilon_distance)

        if close_points < self.empty_threshold:
            self.get_logger().warn('Large empty space detected, possibly end of row.')
            self.stop_robot()

    def stop_robot(self):
        """Stop the robot by publishing a zero velocity command."""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_vel_publisher.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = RowFollowingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

