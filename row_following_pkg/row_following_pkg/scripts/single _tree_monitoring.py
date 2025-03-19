#Astar using /tf for localization (try doing it with the goal position from the ground truth values)
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from tf2_ros import TransformListener, Buffer
from PIL import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Int32


class A_STAR(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* Planner Node Initialized")
        #self.goal_publisher = self.create_publisher(Point, '/astar_goal', 10)
        self.path_coordinates = []

        # TF Buffer and Listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ROS2 Publisher for A* path
        self.path_publisher = self.create_publisher(Path, '/astar_path', 10)

        # Load and process the inflated map
        OG_array = np.array(Image.open('/home/chinmay/second_ros2_ws/src/orch_sim/maps/inflated_map_2.pgm').convert('L')).astype(int)
        self.tree_id_publisher = self.create_publisher(Int32, '/astar_tree_id', 10)        
        OG_array[OG_array <= 128] = 0  # Obstacles
        OG_array[OG_array > 128] = 255  # Free space
        OG_array = np.flipud(OG_array)
        self.occupancy_grid = OG_array
        print("Inflated Map dimensions:", OG_array.shape)

        # Parameters
        self.resolution = 0.05  # Resolution in meters per cell
        self.origin = [-7.3, -0.579]  # Inflated map origin in meters (from map.yaml)

        # Start and goal positions
        self.start_pose = None  # Will be updated from /tf
        self.goal_pose = None  # Set dynamically later
        self.last_start = None
        self.last_goal = None
        self.timer = self.create_timer(2.0, self.run_astar)

        # Load tree database
        self.tree_db_path = '/home/chinmay/second_ros2_ws/src/row_following_pkg/tree_db_2.json'
        self.tree_positions = self.load_tree_positions()

    # def publish_goal_position(self, goal_pose_real):
    #     point_msg = Point()
    #     point_msg.x = goal_pose_real[0]
    #     point_msg.y = goal_pose_real[1]
    #     point_msg.z = 0.0  # Assuming a 2D plane
    #     self.goal_publisher.publish(point_msg)
    #     self.get_logger().info(f"Published Goal Position: {point_msg}")

    def load_tree_positions(self):
        """
        Load tree positions from the JSON database.
        """
        try:
            with open(self.tree_db_path, 'r') as f:
                tree_data = json.load(f)
            return {int(k): tuple(v['tree_pos']) for k, v in tree_data['trees'].items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load tree database: {e}")
            return {}

    def map_to_grid(self, real_x, real_y):
        """
        Transform real-world coordinates (from /map) to grid indices (for A*).
        """
        grid_x = int((real_x - self.origin[0]) / self.resolution)
        grid_y = int((real_y - self.origin[1]) / self.resolution)
        return grid_y, grid_x  # Note: (row, col) for the grid

    def grid_to_map(self, grid_x, grid_y):
        """
        Transform grid indices (from A*) to real-world coordinates (for /map).
        """
        real_x = grid_x * self.resolution + self.origin[0]
        real_y = grid_y * self.resolution + self.origin[1]
        return real_x, real_y

    def update_start_pose_from_tf(self):
        """
        Update the start pose based on the map → odom → base_link transform.
        """
        try:
            # Lookup transform from map to base_link
            transform = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())

            # Extract the translation (position)
            real_x = transform.transform.translation.x
            real_y = transform.transform.translation.y
            
            # Extract the rotation (orientation)
            rotation_q = transform.transform.rotation
            rotation = R.from_quat([rotation_q.x, rotation_q.y, rotation_q.z, rotation_q.w])
            roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)

            # Convert real-world coordinates to grid indices
            new_start_pose = self.map_to_grid(real_x, real_y)
            print(f"Start Pose (Real): ({real_x}, {real_y})")
            print(f"Start Pose (Grid): {new_start_pose}")

            if self.start_pose != new_start_pose:
                self.start_pose = new_start_pose
                self.get_logger().info(f"Updated Start Pose from TF: Real({real_x}, {real_y}), Grid{self.start_pose}")
        except Exception as e:
            self.get_logger().warn(f"Could not get transform: {e}")

    def set_goal_pose(self, tree_id):
        """
        Set the goal position based on the tree ID.
        Moves the goal position to a valid location near the tree.
        """
        if tree_id not in self.tree_positions:
            self.get_logger().error(f"Tree ID {tree_id} not found in database.")
            return
        # Tree position (real-world)
        tree_x, tree_y = self.tree_positions[tree_id]
        # Offset goal position to be near the tree
        offset_distance = 4.0  # Distance from the tree along the y-axis
        offset_x = tree_x  # Keep x-coordinate the same as the tree
        offset_y = tree_y + offset_distance  # Adjust y-coordinate by the offset
        # Transform to grid coordinates
        self.goal_pose = self.map_to_grid(offset_x, offset_y)
        print(f"Goal Pose (Real): ({offset_x}, {offset_y})")
        print(f"Goal Pose (Grid): {self.goal_pose}")
        self.get_logger().info(f"Goal for Tree ID {tree_id}: Real({offset_x}, {offset_y}), Grid{self.goal_pose}")
        print(f"Tree {tree_id}: Goal Pose -> Real({offset_x}, {offset_y}), Grid{self.goal_pose}")

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()
        for point in path:
            # Convert grid indices to real-world coordinates
            real_x, real_y = self.grid_to_map(point[1], point[0])  # Swap (row, col) for (x, y)
            pose = PoseStamped()
            pose.pose.position.x = real_x
            pose.pose.position.y = real_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.path_publisher.publish(path_msg)
        self.get_logger().info("Published A* path")

    def run_astar(self):
        # Update start pose from TF
        self.update_start_pose_from_tf()
        if not self.start_pose or not self.goal_pose:
            self.get_logger().warn("Start or Goal pose not set yet.")
            return

        if hasattr(self, 'last_start') and hasattr(self, 'last_goal'):
            if self.start_pose == self.last_start and self.goal_pose == self.last_goal:
                self.get_logger().info("Start and Goal unchanged. Skipping A*.")
                return
        self.last_start = self.start_pose
        self.last_goal = self.goal_pose
        self.get_logger().info("Running A* search...")
        path = self.A_STAR_SEARCH(self.start_pose, self.goal_pose)
        if path:
            self.get_logger().info("A* Path computed successfully")
            self.publish_path(path)
            self.visualize_path(path)
        else:
            self.get_logger().error("No path found!")

    def A_STAR_SEARCH(self, s, g):
        import heapq
        CostTo = {s: 0}
        EstTotalCost = {s: self.h(s, g)}
        pred = {}
        visited = set()
        Q = []
        heapq.heappush(Q, (EstTotalCost[s], s))
        while Q:
            _, v = heapq.heappop(Q)
            if v in visited:
                continue
            visited.add(v)

            if v == g:
                return self.RecoverPath(pred, s, g)

            for neighbor in self.N(v):
                if neighbor in visited:
                    continue

                tentative_cost = CostTo[v] + self.w(v, neighbor)
                if neighbor not in CostTo or tentative_cost < CostTo[neighbor]:
                    pred[neighbor] = v
                    CostTo[neighbor] = tentative_cost
                    EstTotalCost[neighbor] = tentative_cost + self.h(neighbor, g)
                    heapq.heappush(Q, (EstTotalCost[neighbor], neighbor))
        return []

    def N(self, v):
        """
        Generate valid neighbors for a given cell, respecting the inflated map.
        """
        neighbors = [
            (v[0] + 1, v[1]),
            (v[0] + 1, v[1] + 1),
            (v[0], v[1] + 1),
            (v[0] - 1, v[1] + 1),
            (v[0] - 1, v[1]),
            (v[0] - 1, v[1] - 1),
            (v[0], v[1] - 1),
            (v[0] + 1, v[1] - 1),
        ]

        valid_neighbors = []
        for n in neighbors:
            if (
                0 <= n[0] < self.occupancy_grid.shape[0]
                and 0 <= n[1] < self.occupancy_grid.shape[1]
            ):
                if self.occupancy_grid[n[0]][n[1]] == 255:  # Free space in inflated map
                    valid_neighbors.append(n)
                else:
                    self.get_logger().debug(f"Neighbor {n} is blocked.")
            else:
                self.get_logger().debug(f"Neighbor {n} is out of bounds.")

        return valid_neighbors

    def h(self, v1, v2):
        return math.dist(v1, v2)

    def w(self, v1, v2):
        return self.d(v1, v2)

    def d(self, v1, v2):
        return math.dist(v1, v2)

    def RecoverPath(self, pred, s, g):
        path = [g]
        current = g
        while pred[current] != s:
            path.append(pred[current])
            current = pred[current]
        path.append(s)
        path.reverse()
        return path

    def visualize_path(self, path):
        if not path:
            self.get_logger().error("Cannot visualize path: No path found!")
            return

        X = [self.grid_to_map(point[1], point[0])[0] for point in path]
        Y = [self.grid_to_map(point[1], point[0])[1] for point in path]

        # Plot the occupancy grid
        plt.imshow(
            self.occupancy_grid,
            origin='lower',
            cmap='gray',
            interpolation='none',
            extent=[
                self.origin[0],
                self.origin[0] + self.occupancy_grid.shape[1] * self.resolution,
                self.origin[1],
                self.origin[1] + self.occupancy_grid.shape[0] * self.resolution,
            ]
        )

        # Mark all tree positions with IDs
        for tree_id, (tree_x, tree_y) in self.tree_positions.items():
            plt.scatter(tree_x, tree_y, color='blue', s=20, label="Tree" if tree_id == 1 else "")
            plt.text(tree_x, tree_y, f"{tree_id}", fontsize=8, color='red', ha='center', va='center')

        # Mark the goal tree position
        if self.goal_pose:
            real_goal_x, real_goal_y = self.grid_to_map(self.goal_pose[1], self.goal_pose[0])
            plt.scatter(real_goal_x, real_goal_y, color='orange', s=50, label="Goal Tree")
            plt.text(real_goal_x, real_goal_y, "Goal", fontsize=10, color='orange', ha='center', va='center')

        # Plot the path
        plt.plot(X, Y, '-r', label="A* Path")
        plt.scatter(X[0], Y[0], color='green', label="Start")
        plt.scatter(X[-1], Y[-1], color='orange', label="Goal")
        plt.xlabel("X [meters]")
        plt.ylabel("Y [meters]")
        plt.legend()
        plt.title("A* Path Visualization with Tree IDs")
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = A_STAR()
    node.set_goal_pose(9) # Set Tree ID 93 as the goal
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
