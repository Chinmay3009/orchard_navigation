import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from functools import partial
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# Circle Fitting Function
def fit_circle_to_points(x, y):
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x**2 + y**2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    xc = c[0]
    yc = c[1]
    radius = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, radius

class LidarDBSCANNode(Node):
    def __init__(self, tree_db_path="tree_db.json"):
        super().__init__('lidar_dbscan_node')

        self.lidar_points = []
        self.tree_db_path = tree_db_path
        self.load_tree_db()

        # Subscribe to the LIDAR topic
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)

        # Set up a timer to run DBSCAN every 0.2 seconds
        self.timer = self.create_timer(0.2, self.run_dbscan)
        print("Node initialized and timer started.")

    def load_tree_db(self):
        """ Load existing tree data from JSON, or initialize an empty dict. """
        try:
            with open(self.tree_db_path, 'r') as f:
                self.tree_db = json.load(f)
        except FileNotFoundError:
            self.tree_db = {}
        self.next_tree_id = max(map(int, self.tree_db.keys()), default=-1) + 1
        print("Tree database loaded:", self.tree_db)

    def save_tree_db(self):
        """ Save tree data to JSON file. """
        with open(self.tree_db_path, 'w') as f:
            json.dump(self.tree_db, f)
        print("Tree database saved.")

    def lidar_callback(self, scan_data, lidar_height):
        """ Convert LaserScan to Cartesian coordinates and add height (z). """
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
        print(f"Received {len(points)} points from LiDAR.")

    def run_dbscan(self):
        """ Perform DBSCAN clustering and assign tree IDs. """
        if len(self.lidar_points) == 0:
            print("No points received yet.")
            return
        
        # Concatenate all points from previous callbacks
        all_points = np.vstack(self.lidar_points)
        print(f"Running DBSCAN on {len(all_points)} points.")

        # Run DBSCAN clustering
        dbscan = DBSCAN(eps=0.6, min_samples=13)
        labels = dbscan.fit_predict(all_points)

        # Assign IDs to clusters
        self.assign_tree_ids(all_points, labels)

        # Clear points after processing
        self.lidar_points = []

    def find_nearest_tree_id(self, xc, yc, threshold=0.5):
        """ Find the nearest tree ID based on centroid coordinates. """
        for tree_id, coord in self.tree_db.items():
            dist = np.sqrt((coord[0] - xc)**2 + (coord[1] - yc)**2)
            if dist < threshold:
                return tree_id
        return None

    def assign_tree_ids(self, points, labels):
        """ Assign or reuse IDs for detected clusters. """
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1:
                continue
            cluster_points = points[labels == label]
            xc, yc, radius = fit_circle_to_points(cluster_points[:, 0], cluster_points[:, 1])

            tree_id = self.find_nearest_tree_id(xc, yc)
            if tree_id is None:
                tree_id = str(self.next_tree_id)
                self.tree_db[tree_id] = (xc, yc)
                self.next_tree_id += 1
                print(f"New tree detected at ({xc:.2f}, {yc:.2f}), assigned ID: {tree_id}")
            else:
                print(f"Tree {tree_id} found again at ({xc:.2f}, {yc:.2f}).")

            # Visualize each detected cluster with its assigned tree ID
            self.visualize_tree(cluster_points, tree_id)

        self.save_tree_db()

    def visualize_tree(self, cluster_points, tree_id):
        """ Visualize the tree cluster with its ID. """
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Tree {tree_id}")
        xc, yc, radius = fit_circle_to_points(cluster_points[:, 0], cluster_points[:, 1])
        circle = plt.Circle((xc, yc), radius, color='b', fill=False, linewidth=2)
        
        ax = plt.gca()
        ax.add_artist(circle)
        ax.set_title("DBSCAN Clustering with Fitted Circles (Top-Down View)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()

        plt.pause(0.01)  # Pause briefly to update the plot
        plt.draw()

def main(args=None):
    rclpy.init(args=args)
    node = LidarDBSCANNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
