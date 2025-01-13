import bpy
import json
import matplotlib.pyplot as plt

# Load heatmap data from a JSON file
file_path = '/Users/julian/Documents/PhD/projects/2024-11 -- levisim on GitHub/levisim/notebooks/heatmap_data.json'  # Update to your file location
with open(file_path, 'r') as f:
    heatmap_data = json.load(f)

# Heatmap grid dimensions
# grid_size_x = len(next(iter(heatmap_data.values())))  # Number of rows
# grid_size_y = len(next(iter(heatmap_data.values()))[0])  # Number of columns
grid_size_x = len(next(iter(heatmap_data)))  # Number of rows
grid_size_y = len(next(iter(heatmap_data)))  # Number of columns

print(f'{heatmap_data=}')
print(f'{type(heatmap_data)=}')
print(f'{len(heatmap_data)=}')
print(f'{type(heatmap_data[0])=}')
print(f'{list(heatmap_data[0].keys())=}')
print(f'{len(list(heatmap_data[0].values()))=}')
print(f'{list(heatmap_data[0].values())[0][0]=}')
print(f'{len(list(heatmap_data[0].values())[0])=}')

first_slice = list(heatmap_data[0].values())[0][0]
grid_size_x = len(first_slice)

first_map = list(heatmap_data[0].values())[0]
grid_size_y = len(first_map)

print(f'{grid_size_x=}')
print(f'{grid_size_y=}')

# Frame settings
frame_start = 1
frame_end = len(heatmap_data)
bpy.context.scene.frame_start = frame_start
bpy.context.scene.frame_end = frame_end

# Function to create or update a heatmap grid for a specific frame
def create_heatmap_for_frame(frame, data):
    for x in range(grid_size_x):
        for y in range(grid_size_y):
            print(f'{type(data)=}')
            print(f'{list(data.keys())=}')
            # print(f'{data["0"]}')
            print(f'{len(data["0"])=}')
            print(f'{len(data["0"][0])=}')
            print(f'{x=}')
            value = data[str(frame)][y][x]  # Heatmap intensity at (x, y)
            print(f'{type(value)=}')
            print(f'{value=}')


            # Create or reuse objects for the grid
            obj_name = f'HeatmapCell_{x}_{y}'
            if obj_name in bpy.data.objects:
                obj = bpy.data.objects[obj_name]
                obj.location.z = value  # Update height for the current frame
            else:
                # Replace cube with small sphere
                print(f'{type(x)=}')
                print(f'{type(y)=}')
                print(f'{type(value)=}')
                bpy.ops.mesh.primitive_uv_sphere_add(radius=0.01, location=(0.1*x, 0.1*y, value))
                obj = bpy.context.object
                obj.name = obj_name

            # Assign color to the sphere based on its z-value
            color = plt.cm.viridis(value)[:3]  # RGB only
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

# Iterate through frames and generate the heatmap, updating both position and color
for frame, heatmap in enumerate(heatmap_data):
    create_heatmap_for_frame(frame, heatmap)
