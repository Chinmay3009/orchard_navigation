import cv2
import numpy as np
import os

# Paths for the original map and where to save the inflated map
original_map_path = "/home/chinmay/second_ros2_ws/first_map_save.pgm"
inflated_map_path = "/home/chinmay/second_ros2_ws/src/orch_sim/maps/inflated_map_3.pgm"

# Load the original map in grayscale
original_map = cv2.imread(original_map_path, cv2.IMREAD_GRAYSCALE)
if original_map is None:
    raise FileNotFoundError(f"Original map not found at {original_map_path}")
print(original_map.shape)
# Map metadata
resolution = 0.05  # meters/pixel (example, update with actual value if needed)
origin = [-7.3, -0.579, 0]  # origin in meters (example, update if needed)

# Create a mask for the obstacle areas (where pixel value is 0)
obstacle_mask = (original_map == 0).astype(np.uint8)  # Mask of obstacles as binary

# Inflate the obstacles horizontally using a rectangular kernel
horizontal_kernel_size = (75, 95)  # Adjust width for horizontal inflation
horizontal_kernel = np.ones(horizontal_kernel_size, np.uint8)
inflated_obstacles = cv2.dilate(obstacle_mask, horizontal_kernel, iterations=1)

# Merge the inflated obstacles back into the original map, keeping free and unknown areas
inflated_map = np.where(inflated_obstacles == 1, 0, original_map)

# Ensure save directory exists
save_directory = os.path.dirname(inflated_map_path)
os.makedirs(save_directory, exist_ok=True)

# Save the inflated map to a file
success = cv2.imwrite(inflated_map_path, inflated_map)
if success:
    print(f"Inflated map saved successfully to {inflated_map_path}")
else:
    print("Failed to save the inflated map.")

# Optional: YAML Metadata File (save as 'inflated_map.yaml' in the same directory)
yaml_metadata_path = os.path.join(save_directory, "inflated_map.yaml")
with open(yaml_metadata_path, "w") as yaml_file:
    yaml_file.write(f"""image: {os.path.basename(inflated_map_path)}
resolution: {resolution}
origin: {origin}
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
""")
print(f"YAML metadata saved to {yaml_metadata_path}")

import matplotlib.pyplot as plt

# Display the binary occupancy map of the inflated map
plt.imshow(inflated_map, cmap='gray', origin='upper')
plt.title("Inflated Binary Occupancy Map")
plt.axis('off')  # Hide axes for a cleaner look
plt.show()
