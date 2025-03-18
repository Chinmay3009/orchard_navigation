import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET

# Parse the .world file to extract model (tree) positions
def parse_world_file(world_file):
    tree_positions = []
    
    # Load the .world file from disk
    tree = ET.parse(world_file)
    root = tree.getroot()

    # Find all 'include' tags with a tree model (assuming the model names start with "OakTree")
    for model in root.findall(".//include"):
        name = model.find('name').text if model.find('name') is not None else ""
        if "OakTree" in name:  # Only process oak trees
            pose = model.find('pose').text
            x, y, z = map(float, pose.split()[:3])  # Extract x, y, z coordinates
            tree_positions.append((x, y))

    return tree_positions

# Convert real-world coordinates to pixel coordinates
def world_to_pixel(world_x, world_y, origin_x, origin_y, resolution, map_size):
    pixel_x = int((world_x - origin_x) / resolution)
    pixel_y = int((world_y - origin_y) / resolution)
    return pixel_x, map_size[1] - pixel_y  # Invert y-axis for image coordinates

# Generate a ground truth map (PGM format)
def generate_ground_truth_map(world_file, map_size, resolution, origin, tree_radius):
    # Create an empty (white) map
    ground_truth_map = np.ones(map_size, dtype=np.uint8) * 255
    
    # Parse tree positions from the .world file
    tree_positions = parse_world_file(world_file)

    # Add trees to the map (black circles)
    for world_x, world_y in tree_positions:
        pixel_x, pixel_y = world_to_pixel(world_x, world_y, origin[0], origin[1], resolution, map_size)
        
        # Draw a filled circle for each tree (with a fixed radius)
        rr, cc = np.ogrid[:tree_radius * 2, :tree_radius * 2]
        mask = (rr - tree_radius) ** 2 + (cc - tree_radius) ** 2 <= tree_radius ** 2
        y_slice = slice(pixel_y - tree_radius, pixel_y + tree_radius)
        x_slice = slice(pixel_x - tree_radius, pixel_x + tree_radius)
        if 0 <= pixel_y - tree_radius < map_size[1] and 0 <= pixel_x - tree_radius < map_size[0]:
            ground_truth_map[y_slice, x_slice][mask] = 0  # Set tree to black

    # Save the map as a PGM file
    map_image = Image.fromarray(ground_truth_map)
    map_image.save("ground_truth_map.pgm")

# Main configuration
world_file = './worlds/orch.world'  # Path to your .world file
map_size = (1000, 1000)  # Width, height of the map in pixels
resolution = 0.1  # Meters per pixel
origin = (-10, -10)  # Origin of the map in real-world coordinates (x, y)
tree_radius = 5  # Radius of trees in pixels

# Generate the ground truth map
generate_ground_truth_map(world_file, map_size, resolution, origin, tree_radius)

