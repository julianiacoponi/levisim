import bpy
import json
import matplotlib.pyplot as plt

# Load heatmap data from a JSON file
file_path = '/Users/julian/Downloads/heatmap_3d_translation.json'  # Update to your file location
with open(file_path, 'r') as f:
    heatmap_data = json.load(f)

# Heatmap grid dimensions
grid_size_x = len(next(iter(heatmap_data.values())))  # Number of rows
grid_size_y = len(next(iter(heatmap_data.values()))[0])  # Number of columns

# Frame settings
frame_start = 1
frame_end = len(heatmap_data)
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

# Function to map z values to colors using the viridis colormap
def get_color_from_z(z_value, z_min, z_max):
    # Normalize the z_value to a range between 0 and 1
    normalized_z = (z_value - z_min) / (z_max - z_min)
    # Get the color from the viridis colormap
    color = plt.cm.viridis(normalized_z)  # Get RGBA values
    return color[:3]  # Return RGB only

# Function to create or update a heatmap grid for a specific frame
def create_heatmap_for_frame(frame, data, z_min, z_max):
    for ii_x, x in enumerate(range(grid_size_x)):
        for ii_y, y in enumerate(range(grid_size_y)):
            value = data[x][y]  # Heatmap intensity at (x, y)

            # Create or reuse objects for the grid
            obj_name = f'HeatmapCell_{ii_x}_{ii_y}'
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.location.z = value  # Update height for the current frame
            else:
                # Replace cube with small sphere
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x, y, value))
                obj = bpy.context.object
                obj.name = obj_name

            # Assign color to the sphere based on its z-value
            color = get_color_from_z(value, z_min, z_max)

            # Check if the sphere already has a material; if not, create one
            if not obj.data.materials:
                mat = bpy.data.materials.new(name=f"HeatmapMaterial_{x}_{y}")
                mat.use_nodes = True
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs['Base Color'].default_value = (*color, 1)  # Set RGBA color
                obj.data.materials.append(mat)
            else:
                # If it already has a material, update the color
                mat = obj.data.materials[0]
                bsdf = mat.node_tree.nodes["Principled BSDF"]
                bsdf.inputs['Base Color'].default_value = (*color, 1)  # Set RGBA color

            # Insert keyframe for the location (position)
            obj.keyframe_insert(data_path='location', frame=frame)

            # Insert keyframe for the material color (Base Color)
            mat = obj.data.materials[0]
            bsdf = mat.node_tree.nodes["Principled BSDF"]
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=frame)

# Find the global min and max z values to normalize the color scale
z_values = [value for frame_data in heatmap_data.values() for row in frame_data for value in row]
z_min = min(z_values)
z_max = max(z_values)

# Iterate through frames and generate the heatmap, updating both position and color
for frame, heatmap in heatmap_data.items():
    frame = int(frame)  # Ensure the frame is an integer
    create_heatmap_for_frame(frame, heatmap, z_min, z_max)
