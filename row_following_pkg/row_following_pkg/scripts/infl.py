#path follower for multiple tree monitoring
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
from math import sqrt, atan2, pi
import matplotlib.pyplot as plt
import time

class PathFollowerWithError(Node):
    def __init__(self):
        super().__init__('path_follower_with_error')

        # ROS2 Subscribers and Publisher
        self.create_subscription(Path, '/astar_path', self.path_callback, 10)
        self.create_subscription(Odometry, '/wheel/odometry', self.odometry_callback, 10)
        self.velocity_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # State variables
        self.path = []
        self.current_position = None
        self.current_orientation = None
        self.current_waypoint_index = 0
        self.goal_position = None
        self.goal_reached = False
        self.tree_errors = []
        self.current_tree = 0

        # Control parameters
        self.linear_speed = 0.4
        self.angular_speed_limit = 0.3
        self.reached_threshold = 0.2
        self.lookahead_distance = 0.5
        self.smoothing_factor = 0.2
        self.previous_angular_velocity = 0.0

    def path_callback(self, msg):
        self.get_logger().info("Received new path.")
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        if self.path:
            self.goal_position = self.path[-1]
        self.current_waypoint_index = 0
        self.goal_reached = False

    def odometry_callback(self, msg):
        # Update current position and orientation from odometry
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_position = (x, y)

        orientation_q = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation_q.w * orientation_q.z + orientation_q.x * orientation_q.y)
        cosy_cosp = 1 - 2 * (orientation_q.y**2 + orientation_q.z**2)
        self.current_orientation = atan2(siny_cosp, cosy_cosp)

        self.follow_path()

    def follow_path(self):
        if not self.path or self.goal_reached:
            return

        if not self.current_position or not self.current_orientation:
            self.get_logger().warn("Waiting for odometry data...")
            return

        # Check if robot is close enough to the final goal
        distance_to_goal = sqrt(
            (self.goal_position[0] - self.current_position[0]) ** 2 +
            (self.goal_position[1] - self.current_position[1]) ** 2
        )
        if distance_to_goal <= self.reached_threshold:
            self.get_logger().info("Reached goal. Stopping for 3 seconds.")
            self.stop_robot()
            time.sleep(3)
            self.goal_reached = True

            # Calculate and record error
            goal_x, goal_y = self.goal_position
            final_position_x, final_position_y = self.current_position

            error = sqrt((goal_x - final_position_x) ** 2 + (goal_y - final_position_y) ** 2)
            self.tree_errors.append((self.current_tree, error))
            self.get_logger().info(f"Tree {self.current_tree} error: {error} meters")
            print(f"Tree {self.current_tree}: Goal({goal_x}, {goal_y}) vs Final({final_position_x}, {final_position_y}) -> Error: {error} meters")
            # Move to the next tree
            self.current_tree += 1
            if self.current_tree > 100:
                self.plot_errors()
                self.get_logger().info("Completed all trees. Shutting down.")
                rclpy.shutdown()
            else:
                self.reset_robot()
            return

        # Get lookahead point
        lookahead_point = self.get_lookahead_point()
        if not lookahead_point:
            self.get_logger().info("Lookahead point not found. Stopping the robot.")
            self.stop_robot()
            self.goal_reached = True
            return

        # Calculate control
        dx = lookahead_point[0] - self.current_position[0]
        dy = lookahead_point[1] - self.current_position[1]
        distance_to_lookahead = sqrt(dx ** 2 + dy ** 2)
        angle_to_lookahead = atan2(dy, dx)

        # Compute control commands
        angle_error = self.normalize_angle(angle_to_lookahead - self.current_orientation)
        angular_velocity = 2.0 * angle_error
        angular_velocity = max(-self.angular_speed_limit, min(self.angular_speed_limit, angular_velocity))

        # Apply low-pass filter for smoother angular velocity
        angular_velocity = (
            self.smoothing_factor * angular_velocity
            + (1 - self.smoothing_factor) * self.previous_angular_velocity
        )
        self.previous_angular_velocity = angular_velocity

        linear_velocity = min(self.linear_speed, 0.5 * distance_to_lookahead)

        # Publish velocity commands
        self.publish_velocity(linear_velocity, angular_velocity)

        # if distance_to_goal <= self.reached_threshold:
        #     self.get_logger().info("Reached goal. Stopping for 3 seconds.")
        #     self.stop_robot()
        #     time.sleep(3)
        #     self.goal_reached = True

        #     # Calculate and record error
        #     final_position_x, final_position_y = self.current_position
        #     goal_x, goal_y = self.goal_position

        #     error = sqrt((goal_x - final_position_x) ** 2 + (goal_y - final_position_y) ** 2)
        #     self.tree_errors.append((self.current_tree, error))

        #     self.get_logger().info(f"Final position: ({final_position_x}, {final_position_y})")
        #     self.get_logger().info(f"Goal position: ({goal_x}, {goal_y})")
        #     self.get_logger().info(f"Tree {self.current_tree} error: {error} meters")

        #     # Print the error for manual note-taking
        #     print(f"Tree {self.current_tree}: Goal({goal_x}, {goal_y}) vs Final({final_position_x}, {final_position_y}) -> Error: {error} meters")

    def get_lookahead_point(self):
        if not self.current_position:
            return None

        for i in range(self.current_waypoint_index, len(self.path)):
            waypoint = self.path[i]
            distance = sqrt(
                (waypoint[0] - self.current_position[0])**2 +
                (waypoint[1] - self.current_position[1])**2
            )
            if distance >= self.lookahead_distance:
                self.current_waypoint_index = i
                return waypoint

        return None

    def normalize_angle(self, angle):
        while angle > pi:
            angle -= 2 * pi
        while angle < -pi:
            angle += 2 * pi
        return angle

    def publish_velocity(self, linear, angular):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        self.velocity_publisher.publish(twist)

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.velocity_publisher.publish(twist)

    def reset_robot(self):
        self.get_logger().info(f"Starting path for tree {self.current_tree}.")
        # Reset robot state here if necessary (e.g., move to starting position)
        self.goal_reached = False
        self.current_position = None
        self.current_orientation = None
        self.path = []
        # Trigger new path planning for the next tree

    def plot_errors(self):
        tree_ids = [item[0] for item in self.tree_errors]
        errors = [item[1] for item in self.tree_errors]

        plt.figure()
        plt.plot(tree_ids, errors, marker='o')
        plt.xlabel('Tree Number')
        plt.ylabel('Euclidean Distance Error (meters)')
        plt.title('Path Following Errors for Trees')
        plt.grid()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = PathFollowerWithError()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
