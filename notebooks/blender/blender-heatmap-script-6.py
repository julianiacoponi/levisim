import bpy
import json
import matplotlib.pyplot as plt

# Load heatmap data from a JSON file
file_path = '/Users/julian/Documents/PhD/projects/2024-11 -- levisim on GitHub/levisim/notebooks/heatmap_data.json'  # Update to your file location
with open(file_path, 'r') as f:
    heatmap_data = json.load(f)

# Determine grid dimensions from the heatmap data
first_slice = list(heatmap_data[0].values())[0][0]
grid_size_x = len(first_slice)
first_map = list(heatmap_data[0].values())[0]
grid_size_y = len(first_map)

print(f'{grid_size_x=}, {grid_size_y=}')

# Frame settings
frame_start = 1
frame_end = len(heatmap_data)
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

def create_skyscrapers_for_frame(frame, data):
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            value = data[str(frame)][y][x]  # Heatmap intensity

            obj_name = f'Skyscraper_{x}_{y}'  # Unique name for each bar

            # Check if the object already exists
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.scale.z = value  # Update height (z-scale) based on heatmap
                obj.location.z = value / 2  # Re-center the bar vertically
            else:
                # Create a new skyscraper (cube)
                bpy.ops.mesh.primitive_cube_add(
                    size=1,
                    location=(0.1 * x, 0.1 * y, value / 2)  # Centered at half its height
                )
                obj = bpy.context.object
                obj.name = obj_name
                obj.scale = (0.08, 0.08, value)  # Thin bar in x-y, height = z-value

            # Assign or update the color based on its height (z-value)
            color = plt.cm.viridis(value)[:3]  # Extract RGB from colormap

            # Ensure material exists and has the Principled BSDF node
            if not obj.data.materials:
                mat = bpy.data.materials.new(name=f"SkyscraperMaterial_{x}_{y}")
                mat.use_nodes = True
                obj.data.materials.append(mat)
            else:
                mat = obj.data.materials[0]

            # Get the Principled BSDF node, creating it if necessary
            bsdf = None
            for node in mat.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    bsdf = node
                    break
            if bsdf is None:
                bsdf = mat.node_tree.nodes.new(type='ShaderNodeBsdfPrincipled')

            # Update the Base Color of the material
            bsdf.inputs['Base Color'].default_value = (*color, 1)  # Set RGBA color

            # Insert keyframe for scale and location (height and position)
            obj.keyframe_insert(data_path='scale', frame=frame)
            obj.keyframe_insert(data_path='location', frame=frame)

            # Insert keyframe for the material color
            bsdf.inputs['Base Color'].keyframe_insert(data_path='default_value', frame=frame)


# Iterate through frames and update the skyscrapers
for frame, heatmap in enumerate(heatmap_data):
    create_skyscrapers_for_frame(frame, heatmap)
