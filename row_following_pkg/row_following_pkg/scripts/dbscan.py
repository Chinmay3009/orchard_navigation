import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from functools import partial

class LidarDBSCANNode(Node):
    def __init__(self):
        super().__init__('lidar_dbscan_node')

        # Initialize lidar points as an empty list
        self.lidar_points = []

        # Create subscribers for each LiDAR sensor using partial to pass lidar_height
        #self.create_subscription(LaserScan, '/scan_1', partial(self.lidar_callback, lidar_height=0.0), 10)
        #self.create_subscription(LaserScan, '/scan_2', partial(self.lidar_callback, lidar_height=0.20), 10)
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)


        # Create a timer to run DBSCAN periodically (5Hz = 0.2s)
        self.timer = self.create_timer(0.2, self.run_dbscan)

    def lidar_callback(self, scan_data, lidar_height):
        """ Convert LaserScan to Cartesian coordinates and add height (z) """
        ranges = np.array(scan_data.ranges)
        angles = np.linspace(scan_data.angle_min, scan_data.angle_max, len(ranges))

        # Filter out invalid ranges
        mask = (ranges > scan_data.range_min) & (ranges < scan_data.range_max)
        ranges = ranges[mask]
        angles = angles[mask]

        # Convert polar to Cartesian coordinates
        x_points = ranges * np.cos(angles)
        y_points = ranges * np.sin(angles)
        z_points = np.full_like(x_points, lidar_height)

        # Stack (x, y, z) and store the points
        points = np.column_stack((x_points, y_points, z_points))
        self.lidar_points.append(points)

    def run_dbscan(self):
        """ Perform DBSCAN clustering and visualize the results """
        if len(self.lidar_points) > 0:
            # Concatenate the points from all LiDARs
            all_points = np.vstack(self.lidar_points)

            # Run DBSCAN clustering
            dbscan = DBSCAN(eps=0.8, min_samples=13)  # Tune parameters as needed
            labels = dbscan.fit_predict(all_points)

            # Visualize the clusters
            self.visualize_clusters(all_points, labels)

            # Clear points after processing
            self.lidar_points = []

    def visualize_clusters(self, points, labels):
        """ Visualize raw data and clusters with X-axis as vertical and Y-axis as horizontal """
        unique_labels = set(labels)

        # Create a figure with two subplots side by side
        fig, axes = plt.subplots(1, 2, figsize=(6, 8))  # Two subplots side by side

        # Plot raw data (subplot 1)
        axes[0].scatter(points[:, 1], points[:, 0], c='gray', s=10, label='Raw Data')
        axes[0].set_title("Raw LiDAR Data")
        axes[0].set_xlabel("Y (meters)")  # Y becomes horizontal
        axes[0].set_ylabel("X (meters)")  # X becomes vertical
        axes[0].grid(True)

        # Plot DBSCAN clusters (subplot 2)
        for label in unique_labels:
            if label == -1:
                color = 'k'  # Black for noise
            else:
                color = plt.cm.Spectral(float(label) / len(unique_labels))

            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]

            axes[1].scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], s=10)

        axes[1].set_title("DBSCAN Clustering")
        axes[1].set_xlabel("Y (meters)")  # Y becomes horizontal
        axes[1].set_ylabel("X (meters)")  # X becomes vertical
        axes[1].grid(True)

        plt.tight_layout()
        plt.show()



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
