###A* path planning for multiple tree monitoring
import json
import numpy as np
import math
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from PIL import Image


class A_STAR(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* Planner Node Initialized")
        self.path_coordinates = []

        # ROS2 Publisher for A* path
        self.path_publisher = self.create_publisher(Path, '/astar_path', 10)

        # Subscriber to the odometry to dynamically update the start position
        self.odometry_subscriber = self.create_subscription(
            Odometry,
            '/wheel/odometry',
            self.update_start_pose,
            10
        )

        # Load and process the inflated map
        OG_array = np.array(Image.open('/home/chinmay/second_ros2_ws/src/orch_sim/maps/inflated_map_2.pgm').convert('L')).astype(int)
        OG_array[OG_array <= 128] = 0  # Obstacles
        OG_array[OG_array > 128] = 255  # Free space
        OG_array = np.flipud(OG_array)
        self.occupancy_grid = OG_array
        print("Inflated Map dimensions:", OG_array.shape)

        # Parameters
        self.resolution = 0.05  # Resolution in meters per cell
        self.origin = [-7.3, -0.579]  # Inflated map origin in meters (from map.yaml)

        # Start position
        self.start_pose = None  # Will be updated from /wheel/odometry
        self.last_start = None
        self.last_goal = None

        # Timer for periodic A* planning
        self.timer = self.create_timer(2.0, self.run_astar)

        # Load tree database
        self.tree_db_path = '/home/chinmay/second_ros2_ws/src/row_following_pkg/tree_db.json'
        self.tree_positions = self.load_tree_positions()

        # Tree visitation list
        self.goal_tree_ids = []  # List of tree IDs to visit
        self.visited_trees = set()  # Keep track of visited trees

    def load_tree_positions(self):
        try:
            with open(self.tree_db_path, 'r') as f:
                tree_data = json.load(f)
            return {int(k): tuple(v['tree_pos']) for k, v in tree_data['trees'].items()}
        except Exception as e:
            self.get_logger().error(f"Failed to load tree database: {e}")
            return {}

    def map_to_grid(self, real_x, real_y):
        grid_x = int((real_x - self.origin[0]) / self.resolution)
        grid_y = int((real_y - self.origin[1]) / self.resolution)
        return grid_y, grid_x

    def grid_to_map(self, grid_x, grid_y):
        real_x = grid_x * self.resolution + self.origin[0]
        real_y = grid_y * self.resolution + self.origin[1]
        return real_x, real_y

    def update_start_pose(self, msg):
        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        self.start_pose = self.map_to_grid(real_x, real_y)

    def set_goal_tree_ids(self, tree_ids):
        """
        Set the list of tree IDs to visit and initialize the visited set.
        """
        # Ensure tree_ids is a list of integers or valid tree ID types
        if not isinstance(tree_ids, list):
            raise TypeError(f"Expected a list of tree IDs, got {type(tree_ids)} instead.")

        # Validate that all tree IDs exist in self.tree_positions
        invalid_tree_ids = [tree_id for tree_id in tree_ids if tree_id not in self.tree_positions]
        if invalid_tree_ids:
            raise ValueError(f"The following tree IDs are invalid: {invalid_tree_ids}")

        # Store the list of tree IDs and reset visited set
        self.goal_tree_ids = tree_ids
        self.visited_trees = set()
        self.get_logger().info(f"Set goal tree IDs: {tree_ids}")

    def is_valid_goal(self, goal_grid):
        """
        Check if a grid cell is a valid goal position (free space).
        """
        grid_y, grid_x = goal_grid  # Unpack grid indices
        if (
            0 <= grid_y < self.occupancy_grid.shape[0]
            and 0 <= grid_x < self.occupancy_grid.shape[1]
        ):
            return self.occupancy_grid[grid_y, grid_x] == 255  # Check if free space
        return False

    def run_astar(self):
        if not self.start_pose:
            self.get_logger().warn("Start pose not set yet.")
            return

        if not self.goal_tree_ids:
            self.get_logger().warn("No tree goals provided.")
            return

        self.get_logger().info("Starting multi-tree A* planning...")

        current_position = self.start_pose
        overall_path = []

        while len(self.visited_trees) < len(self.goal_tree_ids):
            closest_tree = None
            shortest_distance = float('inf')

            for tree_id in self.goal_tree_ids:
                if tree_id in self.visited_trees:
                    continue

                tree_position = self.tree_positions[tree_id]
                grid_tree_position = self.map_to_grid(tree_position[0], tree_position[1])
                distance = self.h(current_position, grid_tree_position)
                if distance < shortest_distance:
                    shortest_distance = distance
                    closest_tree = tree_id

            if closest_tree is None:
                break

            self.get_logger().info(f"Planning path to tree {closest_tree}...")
            tree_position = self.tree_positions[closest_tree]
            goal_grid = self.map_to_grid(tree_position[0], tree_position[1])

            # Attempt to find a valid goal position with dynamic offsets
            valid_goal = self.find_valid_goal(goal_grid)
            if not valid_goal:
                self.get_logger().error(f"Could not find a valid goal for tree {closest_tree}. Skipping...")
                self.visited_trees.add(closest_tree)
                continue

            self.goal_pose = valid_goal

            path = self.A_STAR_SEARCH(current_position, self.goal_pose)
            if not path:
                self.get_logger().error(f"No path found to tree {closest_tree}. Skipping...")
                self.visited_trees.add(closest_tree)
                continue

            overall_path.extend(path)
            current_position = self.goal_pose
            self.visited_trees.add(closest_tree)

        if overall_path:
            self.publish_path(overall_path)
            self.visualize_path(overall_path)
        else:
            self.get_logger().error("Failed to compute a valid path for the given trees.")

    def find_valid_goal(self, goal_grid, max_attempts=5, offset_step=2.0):
        """
        Dynamically find a valid goal position near the tree.
        Args:
            goal_grid: The initial grid position of the tree (grid_y, grid_x).
            max_attempts: The maximum number of attempts to find a valid goal.
            offset_step: The step size (in meters) for trying offsets.
        Returns:
            A valid goal grid position or None if no valid position is found.
        """
        grid_y, grid_x = goal_grid
        for attempt in range(1, max_attempts + 1):
            offsets = [
                (grid_y + int(attempt * offset_step / self.resolution), grid_x),  # Positive Y
                (grid_y - int(attempt * offset_step / self.resolution), grid_x),  # Negative Y
                (grid_y, grid_x + int(attempt * offset_step / self.resolution)),  # Positive X
                (grid_y, grid_x - int(attempt * offset_step / self.resolution)),  # Negative X
            ]
            for offset in offsets:
                if self.is_valid_goal(offset):
                    self.get_logger().info(f"Found valid goal at offset {offset} after {attempt} attempts.")
                    return offset

            self.get_logger().warn(f"Attempt {attempt}: No valid goal found near {goal_grid}.")

        self.get_logger().error(f"Exceeded max attempts to find a valid goal near {goal_grid}.")
        return None


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
                if self.occupancy_grid[n[0]][n[1]] == 255:
                    valid_neighbors.append(n)

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

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path:
            real_x, real_y = self.grid_to_map(point[1], point[0])
            pose = PoseStamped()
            pose.pose.position.x = real_x
            pose.pose.position.y = real_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        self.get_logger().info("Published A* path")

    def visualize_path(self, path):
        """
        Visualize the planned path and annotate the waypoints with tree IDs.
        """
        X = [self.grid_to_map(point[1], point[0])[0] for point in path]
        Y = [self.grid_to_map(point[1], point[0])[1] for point in path]

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
        plt.plot(X, Y, '-r', label="A* Path")

        # Annotate waypoints with tree IDs
        for tree_id in self.goal_tree_ids:
            tree_pos = self.tree_positions[tree_id]
            plt.scatter(tree_pos[0], tree_pos[1], color='blue', s=50, label=f"Tree {tree_id}")
            plt.text(tree_pos[0], tree_pos[1], f"Tree {tree_id}", fontsize=8, color='green')
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

    # Example: Provide multiple tree IDs to visit
    node.set_goal_tree_ids([66, 29, 73, 15])

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()