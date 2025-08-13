import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
import json

import sys
sys.path.append('/home/jamba/research/thesis-v2/')

from geometry.geometry_samples import generate_shapes, plot_all_shapes
from toolpath.path_gen import generate_toolpath, calculate_toolpath_stats
from toolpath.fields import calculate_signed_distance_field
from solver.mesher import mesh_polygon
from solver.mesh_utils import get_mesh_statistics_from_gmsh
from solver.utils import save_data, load_data
from geometry.geometry_samples import generate_grid_points
from solver.config import get_mesh_size_presets

def preprocess_input_fields(polygon, toolpath, 
                           all_points=None,
                           sdf_field=None,
                           discretization_length=None):
    """
    Preprocess input fields for temperature prediction.
    
    Args:
        polygon: Shapely Polygon representing the part geometry
        toolpath: Numpy array of shape (n_points, 3) with [time, x, y] coordinates
        all_points: Pre-computed grid points (optional)
        sdf_field: Pre-computed SDF field (optional)
    
    Returns:
        tuple: (raw_fields, inside_mask, all_points, grid_size)
            - raw_fields: List of raw input fields [sdf, time, grad]
            - inside_mask: Boolean mask for points inside the polygon
            - all_points: All grid points with inside/outside information
            - grid_size: Size of the grid (grid_size x grid_size)
    """
    
    # Get mesh size configuration for discretization
    if discretization_length is None:
        mesh_config = get_mesh_size_presets('fine')
        discretization_length = mesh_config['laser']
    
    # Generate evaluation points (grid points within the shape)
    if all_points is None:
        all_points = generate_grid_points(polygon, discretization_length)
    grid_points = all_points[:, :2]
    grid_size = int(np.sqrt(len(grid_points)))
    
    # Check minimum grid size for U-Net
    if grid_size < 8:
        raise ValueError(f"Grid size {grid_size} is too small for U-Net. Minimum size is 8x8.")
    
    # Calculate SDF field
    if sdf_field is None:
        sdf_values = calculate_signed_distance_field(grid_points, polygon)
        sdf_field = sdf_values.reshape((grid_size, grid_size))
    
    # Calculate time field using RBF interpolation
    toolpath_points = toolpath[:, 1:3]
    toolpath_times = toolpath[:, 0]
    
    # Use RBF interpolation for smoother time field
    rbf_interp = RBFInterpolator(toolpath_points, toolpath_times, kernel='linear')
    time_values = rbf_interp(grid_points)
    time_field = time_values.reshape((grid_size, grid_size))
    
    # Calculate gradient magnitude
    grad_y, grad_x = np.gradient(time_field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create inside mask
    inside_mask = all_points[:, 2].reshape((grid_size, grid_size))
    inside_mask = inside_mask.astype(bool)
    
    # Prepare raw fields for model input
    raw_fields = [sdf_field, time_field, grad_mag]
    
    return raw_fields, inside_mask, all_points, grid_size

def get_shapes_from_JSON(json_file, indices=None):
    """
    Get shape geometries (as WKT) from a JSON file in the format of filtered_shapes.json.
    Args:
        json_file (str): Path to the JSON file (e.g., geometry/pa_control/filtered_shapes.json).
        indices (list of int or None): Indices of shapes to extract (these correspond to the 'index' field in each shape dict).
            If None, all shapes are returned.
    Returns:
        list of shapely.geometry.Polygon: List of Polygon objects for the selected shapes.
    """
    import json
    from shapely import wkt
    with open(json_file, 'r') as f:
        data = json.load(f)
    shapes = data["shapes"]
    if indices is None:
        # Return all shapes
        return [wkt.loads(shape["wkt"]) for shape in shapes]
    else:
        # Build a mapping from the 'index' field to the shape dict
        index_to_shape = {shape["index"]: shape for shape in shapes}
        # Extract shapes in the order of the provided indices
        return [wkt.loads(index_to_shape[i]["wkt"]) for i in indices]

if __name__ == "__main__":

    data_parent_dir = os.path.join("/mnt/c/Users/jamba/sim_data/", "DATABASE_OPT")
    os.makedirs(data_parent_dir, exist_ok=True)
    
    num_shapes = 400
    mesh_sizes = 'fine'
    # Get mesh size configuration for discretization
    mesh_config = get_mesh_size_presets(mesh_sizes)
    discretization_length = mesh_config['laser']  # Use part mesh size for evaluation points
    # shapes = generate_shapes(
    #     num_shapes=num_shapes,
    #     num_union_shapes=None,
    #     num_subtract_shapes=None,
    #     hollowness_bias=0.25,
    #     max_attempts=50,
    #     min_area=3e-6
    #     # hole=True
    # )
    # indices = [291, 310, 393, 377, 449]
    indices = [291, 393, 449]
    shapes = get_shapes_from_JSON(os.path.join("geometry", "pa_control", "filtered_shapes.json"), indices)
    
    plot_all_shapes(shapes)
    angles = np.linspace(0, 360, 10, endpoint=False)
    print(angles)

    for i in range(len(shapes)):
        


        mesh_file = mesh_polygon(shapes[i], mesh_sizes=mesh_sizes, mesh_filename=f"mesh_shape_{i:03d}.msh", output_dir=data_parent_dir)
        mesh_stats = get_mesh_statistics_from_gmsh(mesh_file)   
        
        for angle in angles:

            data_dir = os.path.join(data_parent_dir, f"DATA_{i:03d}_{angle:.0f}")
            # Skip processing if this shape's data already exists
            if os.path.exists(data_dir):
                print(f'Skipping shape {i}: directory already exists')
                continue
                
            # Create directory for this shape's data
            os.makedirs(data_dir, exist_ok=True)
            
            # Generate toolpath for the current shape (raster angle 0, uniraster pattern)
            toolpath = generate_toolpath(shapes[i], raster_angle=angle, pattern="uniraster")
            toolpath_stats = calculate_toolpath_stats(toolpath)
            toolpath_stats['raster_angle_degrees'] = angle
            
            # Preprocess input fields (SDF, time, gradient) for the current shape and toolpat
            if i == 0:
                raw_fields, _, all_points, _ = preprocess_input_fields(shapes[i], toolpath)
            else:
                raw_fields, _, all_points, _ = preprocess_input_fields(shapes[i], toolpath, all_points, sdf_field)
            sdf_field, time_field, grad_mag = raw_fields
            
            # Organize fields into a dictionary for saving
            fields = {
                "sdf_field": sdf_field,
                "time_field": time_field,
                "grad_mag": grad_mag,
            }
            
            print('Saving data for shape: ', i)
            # Save all relevant data for this shape to disk
            save_data(output_dir=data_dir, 
                    shape=shapes[i], 
                    toolpath=toolpath, 
                    toolpath_stats=toolpath_stats, 
                    mesh_file=mesh_file, 
                    fields=fields, 
                    mesh_stats=mesh_stats,
                    inside_outside_array=all_points,
                    output_name="metadata_00")
    
    # load data
    data = os.path.join(data_parent_dir, "DATA_000")
    toolpath, fields, metadata, inside_outside = load_data(data, output_name="metadata_00")
    # plot_toolpath(toolpath)

    size = int(np.sqrt(len(inside_outside)))
    plt.imshow(inside_outside[:, 2].reshape(size, size), cmap='hot')
    plt.title('Evaluation Points')
    plt.axis('equal')
    plt.show() 