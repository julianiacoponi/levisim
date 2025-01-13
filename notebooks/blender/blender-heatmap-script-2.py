import bpy
import json

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

# Function to create a heatmap grid for a specific frame
def create_heatmap_for_frame(frame, data):
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            value = data[x][y]  # Heatmap intensity at (x, y)

            # Create or reuse objects for the grid
            obj_name = f'HeatmapCell_{x}_{y}'
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.location.z = value  # Update height for the current frame
            else:
                # Replace cube with small sphere
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=(x, y, value))
                obj = bpy.context.object
                obj.name = obj_name

            # Insert keyframe for animation
            obj.keyframe_insert(data_path='location', frame=frame)

# Iterate through frames and generate the heatmap
for frame, heatmap in heatmap_data.items():
    frame = int(frame)  # Ensure the frame is an integer
    create_heatmap_for_frame(frame, heatmap)
