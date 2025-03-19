import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from functools import partial

# Function to fit a robust line using RANSAC
def fit_robust_line(x, y):
    ransac = RANSACRegressor()
    x = np.array(x).reshape(-1, 1)
    y = np.array(y)
    ransac.fit(x, y)
    line_coeff = (ransac.estimator_.coef_[0], ransac.estimator_.intercept_)
    return line_coeff

# Function to plot a line
def plot_line(ax, coeffs, x_range, color='r'):
    x_vals = np.linspace(x_range[0], x_range[1], 100)
    y_vals = coeffs[0] * x_vals + coeffs[1]
    ax.plot(x_vals, y_vals, color=color, linewidth=2)

# Main Class for the Node
class LidarDBSCANNode(Node):
    def __init__(self):
        super().__init__('lidar_dbscan_node')

        # Initialize lidar points as an empty list
        self.lidar_points = []

        # Create subscribers for each LiDAR sensor (fix: queue size set to 10)
        self.create_subscription(LaserScan, '/scan_1', partial(self.lidar_callback, lidar_height=0.0), 10)
        self.create_subscription(LaserScan, '/scan_2', partial(self.lidar_callback, lidar_height=0.20), 10)
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
        """ Perform DBSCAN clustering, fit lines, and fit circles """
        if len(self.lidar_points) > 0:
            # Concatenate the points from all LiDARs
            all_points = np.vstack(self.lidar_points)

            # Run DBSCAN clustering
            dbscan = DBSCAN(eps=0.6, min_samples=13)  # Tune parameters as needed
            labels = dbscan.fit_predict(all_points)

            # Visualize the results
            self.visualize_results(all_points, labels)

            # Clear points after processing
            self.lidar_points = []

    def visualize_results(self, points, labels):
        """ Visualize raw data, DBSCAN clusters, and circle fitting in a horizontal layout with rotation """
        unique_labels = set(labels)

        # Determine the coordinate limits for all plots
        x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
        y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])

        # Create a figure with 3 subplots in one row
        fig, axs = plt.subplots(1, 3, figsize=(6, 4))  # Adjusted for horizontal layout

        # Raw LiDAR Data (Left)
        axs[0].scatter(points[:, 1], points[:, 0], c='gray', s=10)  # Swap axes for 90-degree rotation
        axs[0].set_title("LiDAR", fontsize=16)
        #axs[0].set_xlabel("Y (meters)", fontsize=14)  # Adjusted for rotated axes
        axs[0].set_ylabel("X (meters)", fontsize=14)  # Adjusted for rotated axes
        axs[0].set_xlim(y_min, y_max)
        axs[0].set_ylim(x_min, x_max)
        axs[0].grid(True, linestyle='--', alpha=0.6)

        # DBSCAN Clustering (Middle)
        for label in unique_labels:
            if label == -1:
                color = 'k'  # Black for noise
            else:
                color = plt.cm.Spectral(float(label) / len(unique_labels))
            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]
            axs[1].scatter(cluster_points[:, 1], cluster_points[:, 0], c=[color], s=10)  # Swap axes
        axs[1].set_title("DBSCAN", fontsize=16)
        axs[1].set_xlabel("Y (meters)", fontsize=14)  # Adjusted for rotated axes
        #axs[1].set_ylabel("X (meters)", fontsize=14)  # Adjusted for rotated axes
        axs[1].set_xlim(y_min, y_max)
        axs[1].set_ylim(x_min, x_max)
        axs[1].grid(True, linestyle='--', alpha=0.6)

        # Circle Fitting (Right)
        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            class_member_mask = (labels == label)
            cluster_points = points[class_member_mask]
            x_mean = np.mean(cluster_points[:, 0])
            y_mean = np.mean(cluster_points[:, 1])
            circle = plt.Circle((y_mean, x_mean), 1.5, color='blue', fill=False, linewidth=2)  # Swap axes for circle
            axs[2].add_artist(circle)
            axs[2].scatter(cluster_points[:, 1], cluster_points[:, 0], s=10)  # Swap axes
        axs[2].set_title("Circle Fitting", fontsize=16)
        #axs[2].set_xlabel("Y (meters)", fontsize=14)  # Adjusted for rotated axes
        #axs[2].set_ylabel("X (meters)", fontsize=14)  # Adjusted for rotated axes
        axs[2].set_xlim(y_min, y_max)
        axs[2].set_ylim(x_min, x_max)
        axs[2].grid(True, linestyle='--', alpha=0.6)

        # Adjust layout for better spacing
        plt.tight_layout()

        # Save the figure as a high-resolution image for the report
        plt.savefig("lidar_analysis_horizontal.png", dpi=300, bbox_inches='tight')

        # Show the plot
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
