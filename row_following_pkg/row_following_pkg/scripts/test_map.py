##BINARY OCCUPANCY GRID MAP
import rclpy
from nav_msgs.msg import OccupancyGrid
import numpy as np
from roboticstoolbox import BinaryOccupancyGrid
import matplotlib.pyplot as plt

def map_callback(msg):
    # Extract the map data and reshape it into a 2D NumPy array
    width = msg.info.width
    height = msg.info.height
    map_data = np.array(msg.data, dtype=np.int8).reshape((height, width))

    # Create a BinaryOccupancyGrid from the map data
    binary_occupancy_grid = BinaryOccupancyGrid(map_data, size=0.05)
    print((binary_occupancy_grid))
    # Visualize the Binary Occupancy Grid using matplotlib
    plt.imshow(binary_occupancy_grid.grid, origin='lower')
    plt.colorbar(label="Occupancy")
    plt.title("Binary Occupancy Grid Visualization")
    plt.xlabel("X (grid cells)")
    plt.ylabel("Y (grid cells)")
    plt.show()

def main(args=None):
    rclpy.init(args=args)
    node = rclpy.create_node('map_to_numpy')
    node.create_subscription(OccupancyGrid, '/map', map_callback, 10)
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
