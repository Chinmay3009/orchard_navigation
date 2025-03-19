#####MArk tree id
import numpy as np
from sklearn.cluster import DBSCAN
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped
import tf2_ros
from tf2_ros import TransformException
from functools import partial
import tf2_geometry_msgs

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
    def __init__(self, tree_db_path="tree_db_2.json"):  #tree_db_missing.json
        super().__init__('lidar_dbscan_node')

        # Initialize point storage, tree database, tf buffer, and counters
        self.lidar_points = []
        self.tree_db_path = tree_db_path
        self.row_count = 1           # Counter for row number in filenames
        self.next_tree_id = 0         # Start with 0, continue across rows
        self.tree_save_count = 0      # Count of new trees since last save
        self.load_tree_db()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Set up publisher for RViz markers
        self.marker_pub = self.create_publisher(MarkerArray, '/tree_markers', 10)

        # Subscribe to the LIDAR topic
        self.create_subscription(LaserScan, '/scan_3', partial(self.lidar_callback, lidar_height=0.30), 10)

        # Set up a timer to run DBSCAN every 0.2 seconds
        self.timer = self.create_timer(0.2, self.run_dbscan)
        print("Node initialized and timer started.")

    def load_tree_db(self):
        """ Load existing tree data from JSON, or initialize an empty dict. """
        clear_db_on_start = True  # Set this to True to reset database each run
        if clear_db_on_start:
            self.tree_db = {}
            print("Tree database cleared.")
        else:
            try:
                with open(self.tree_db_path, 'r') as f:
                    self.tree_db = json.load(f)
                    # Set `next_tree_id` based on the highest ID in the loaded file
                    self.next_tree_id = max(map(int, self.tree_db.keys()), default=-1) + 1
            except FileNotFoundError:
                self.tree_db = {}
            print("Tree database loaded:", self.tree_db)

    def save_tree_db(self):
        """ Save tree data to JSON file in the desired format. """
        # Convert tree_db structure to match the desired format
        formatted_tree_db = {"trees": {}}
        for tree_id, coords in self.tree_db.items():
            formatted_tree_db["trees"][tree_id] = {"tree_pos": list(coords)}

        with open(self.tree_db_path, 'w') as f:
            json.dump(formatted_tree_db, f, indent=2)
        #print("Tree database saved in desired format.")

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

        # Transform points to the map frame
        transformed_points = []
        for point in points:
            point_stamped = PointStamped()
            point_stamped.header.frame_id = scan_data.header.frame_id
            point_stamped.point.x = point[0]
            point_stamped.point.y = point[1]
            point_stamped.point.z = point[2]
            
            try:
                # Transform point to map frame
                transform = self.tf_buffer.lookup_transform("map", scan_data.header.frame_id, rclpy.time.Time())
                transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)

                transformed_points.append([transformed_point.point.x, transformed_point.point.y, transformed_point.point.z])
            except TransformException as e:
                self.get_logger().error(f"Transform error: {e}")

        if transformed_points:
            self.lidar_points.append(np.array(transformed_points))
        print(f"Received {len(points)} points from LiDAR.")

    def run_dbscan(self):
        """ Perform DBSCAN clustering and assign tree IDs. """
        if len(self.lidar_points) == 0:
            print("No points received yet.")
            return

        all_points = np.vstack(self.lidar_points)
        print(f"Running DBSCAN on {len(all_points)} points.")

        dbscan = DBSCAN(eps=0.6, min_samples=13)
        labels = dbscan.fit_predict(all_points)

        self.assign_tree_ids(all_points, labels)
        self.lidar_points = []

    def find_nearest_tree_id(self, xc, yc, threshold=1.0):
        """ Find the nearest tree ID based on centroid coordinates. """
        for tree_id, coord in self.tree_db.items():
            dist = np.sqrt((coord[0] - xc)**2 + (coord[1] - yc)**2)
            if dist < threshold:
                return tree_id
        return None

    def assign_tree_ids(self, points, labels):
        """ Assign or reuse IDs for detected clusters and publish RViz markers. """
        marker_array = MarkerArray()
        unique_labels = set(labels)
        proximity_threshold = 2.0  # Distance threshold to consider two trees the same

        for label in unique_labels:
            if label == -1:
                continue

            cluster_points = points[labels == label]
            xc, yc, radius = fit_circle_to_points(cluster_points[:, 0], cluster_points[:, 1])

            # Transform xc, yc to the robot's base_link frame to determine right-side position
            try:
                point_stamped = PointStamped()
                point_stamped.header.frame_id = "map"
                point_stamped.point.x = xc
                point_stamped.point.y = yc
                point_stamped.point.z = 0.0
                point_in_robot_frame = self.tf_buffer.transform(point_stamped, "base_footprint", timeout=rclpy.time.Duration(seconds=1.0))

                # Skip trees that are not on the right side of the robot
                if point_in_robot_frame.point.y >= 0:
                    continue

                # Check if this detected tree is close to an existing one
                tree_id = self.find_nearest_tree_id(xc, yc, threshold=proximity_threshold)
                if tree_id is None:
                    # New tree detected
                    tree_id = str(self.next_tree_id)
                    self.tree_db[tree_id] = (xc, yc)
                    self.next_tree_id += 1
                    self.tree_save_count += 1  # Increment count of trees detected since last save
                    print(f"New tree detected at ({xc:.2f}, {yc:.2f}), assigned ID: {tree_id}")
                else:
                    # Update position by averaging to stabilize the location
                    existing_x, existing_y = self.tree_db[tree_id]
                    new_x = (existing_x + xc) / 2
                    new_y = (existing_y + yc) / 2
                    self.tree_db[tree_id] = (new_x, new_y)
                    print(f"Tree {tree_id} found again at approximately ({new_x:.2f}, {new_y:.2f}).")
                    continue  # Skip creating a new marker if it’s a duplicate

                # Create a marker for this tree
                text_marker = Marker()
                text_marker.header.frame_id = "map"
                text_marker.header.stamp = self.get_clock().now().to_msg()
                text_marker.ns = "tree_markers"
                text_marker.id = int(tree_id)
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = xc
                text_marker.pose.position.y = yc
                text_marker.pose.position.z = 1.0
                text_marker.scale.z = 0.5
                text_marker.color.a = 1.0
                text_marker.color.r = 0.0
                text_marker.color.g = 1.0
                text_marker.color.b = 0.0
                text_marker.text = f"Tree {tree_id}"
                marker_array.markers.append(text_marker)

            except TransformException as e:
                self.get_logger().error(f"Transform error: {e}")

        # Publish the marker array to RViz
        self.marker_pub.publish(marker_array)
        print("Markers published to RViz.")
        
        # Save tree database every 10 new trees
        if self.tree_save_count >= 10:
            self.save_tree_db()
            #self.tree_save_count = 0  # Reset the counter


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
