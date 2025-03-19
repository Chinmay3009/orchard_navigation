import numpy as np
import math
import matplotlib.pyplot as plt
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path, OccupancyGrid, Odometry
from geometry_msgs.msg import PoseStamped

class A_STAR(Node):
    def __init__(self):
        super().__init__('astar_planner')
        self.get_logger().info("A* Planner Node Initialized")

        # ROS2 Publisher for A* path
        self.path_publisher = self.create_publisher(Path, '/astar_path', 10)

        # Subscriber to odometry to dynamically update the start position
        self.odometry_subscriber = self.create_subscription(
            Odometry, '/wheel/odometry', self.update_start_pose, 10)

        # Subscriber to the inflated map topic
        self.inflated_map_subscriber = self.create_subscription(
            OccupancyGrid, '/inflated_map', self.update_inflated_map, 10)

        # Parameters
        self.resolution = 0.05  # Resolution in meters per cell
        self.origin = [-7.3, -0.579] # Origin of the map
        self.occupancy_grid = None  # Occupancy grid to be updated dynamically
        self.start_pose = None  # Start position in grid indices
        self.goal_pose = None  # Goal position in grid indices
        self.last_start = None
        self.last_goal = None

        # Timer for running A* periodically
        self.timer = self.create_timer(2.0, self.run_astar)

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

    def update_start_pose(self, msg):
        real_x = msg.pose.pose.position.x
        real_y = msg.pose.pose.position.y
        new_start_pose = self.map_to_grid(real_x, real_y)

        if self.start_pose != new_start_pose:
            self.start_pose = new_start_pose
            self.get_logger().info(f"Updated Start Pose: Real({real_x}, {real_y}), Grid{self.start_pose}")

    def set_goal_pose(self, real_x, real_y):
        """
        Set the goal position based on real-world coordinates (from /map).
        Converts to grid indices for A*.
        """
        self.goal_pose = self.map_to_grid(real_x, real_y)
        self.get_logger().info(f"Real Goal Pose: x={real_x}, y={real_y}")
        self.get_logger().info(f"Grid Goal Pose: {self.goal_pose}")

    def update_inflated_map(self, msg):
        """
        Callback to update the occupancy grid from the inflated map topic.
        """
        width = msg.info.width
        height = msg.info.height
        self.resolution = msg.info.resolution
        self.origin = [msg.info.origin.position.x, msg.info.origin.position.y]

        # Convert the occupancy grid data to a NumPy array
        map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

        # Normalize the values (assuming -1 is unknown, 0 is free, and 100 is occupied)
        map_data[map_data == -1] = 0  # Treat unknown as free
        map_data[map_data == 100] = 255  # Inflate obstacles

        # Update the occupancy grid
        self.occupancy_grid = map_data
        self.get_logger().info(f"Updated occupancy grid from inflated map. Dimensions: {self.occupancy_grid.shape}")

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
        return valid_neighbors

    def d(self, v1, v2):
        return math.dist(v1, v2)

    def w(self, v1, v2):
        return self.d(v1, v2)

    def h(self, v1, v2):
        return self.d(v1, v2)

    def RecoverPath(self, pred, s, g):
        path = [g]
        current = g
        while pred[current] != s:
            path.append(pred[current])
            current = pred[current]
        path.append(s)
        path.reverse()
        return path

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

    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.frame_id = 'map'
        path_msg.header.stamp = self.get_clock().now().to_msg()

        for point in path:
            real_x, real_y = self.grid_to_map(point[1], point[0])  # Swap (row, col) for (x, y)
            pose = PoseStamped()
            pose.pose.position.x = real_x
            pose.pose.position.y = real_y
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)

        self.path_publisher.publish(path_msg)
        self.get_logger().info("Published A* path")

    def visualize_path(self, path):
        if not path:
            self.get_logger().error("Cannot visualize path: No path found!")
            return

        X = [self.grid_to_map(point[1], point[0])[0] for point in path]  # Real-world x
        Y = [self.grid_to_map(point[1], point[0])[1] for point in path]  # Real-world y

        plt.imshow(
            self.occupancy_grid,
            origin='upper',
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
        plt.scatter(X[0], Y[0], color='green', label="Start", zorder=5)
        plt.scatter(X[-1], Y[-1], color='blue', label="Goal", zorder=5)
        plt.title("A* Path Visualization")
        plt.xlabel("X (meters)")
        plt.ylabel("Y (meters)")
        plt.legend()
        plt.show()

    def run_astar(self):
        if self.occupancy_grid is None:
            self.get_logger().warn("Occupancy grid not received yet. Waiting for map data...")
            return
        if not self.start_pose or not self.goal_pose:
            self.get_logger().warn("Start or Goal pose not set yet.")
            return
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


def main(args=None):
    rclpy.init(args=args)
    node = A_STAR()

    # Example: Set a goal in /map real-world coordinates
    node.set_goal_pose(15.0, 3.0)  # Replace with desired goal coordinates

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
