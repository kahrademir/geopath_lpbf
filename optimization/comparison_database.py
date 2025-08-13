import os
import sys
import numpy as np
import json
from shapely import wkt
import gmsh

import torch

# Add parent directory to sys.path to allow importing geometry and toolpath modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from mLearning.models.cnn_12 import myModel
from geometry.geometry_samples import generate_grid_points
from toolpath.path_gen import generate_toolpath, calculate_toolpath_stats
from toolpath.fields import calculate_signed_distance_field
from solver.mesher import mesh_polygon
from solver.mesh_utils import get_mesh_statistics_from_gmsh
from solver.utils import save_data
from solver.config import get_mesh_size_presets
from mLearning.use_model import preprocess_input_fields, predict_temperature_field

def sample_angles_from_optimization(optimization_dir, num_samples=10):
    """
    Load optimization summary and sample num_samples angles (including best and worst).
    """
    summary_path = os.path.join(optimization_dir, "optimization_summary.json")
    with open(summary_path, "r") as f:
        summary = json.load(f)
    angles = np.array(summary["angles"])
    objective_values = np.array(summary["objective_values"])

    # Always include best and worst
    best_idx = np.argmin(objective_values)
    worst_idx = np.argmax(objective_values)
    selected_indices = set([best_idx, worst_idx])

    # Uniformly sample remaining angles
    if len(angles) > 2 and num_samples > 2:
        remaining = [i for i in range(len(angles)) if i not in selected_indices]
        if len(remaining) >= (num_samples - 2):
            step = max(1, len(remaining) // (num_samples - 2))
            sampled = remaining[::step][:num_samples-2]
        else:
            sampled = remaining
        selected_indices.update(sampled)
    selected_indices = sorted(list(selected_indices))
    return angles[selected_indices], selected_indices

def generate_comparison_database(
    optimization_dir,
    output_database_dir,
    mesh_sizes='fine',
    num_samples=10
    ):
    """
    Generate a DATABASE for comparing best/worst/other angles from optimization with simulation data.
    """

    # Load model
    model = myModel(in_channels=2, out_channels=1, bilinear=True, kernel_size=7).to('cuda')
    model_path = "/home/jamba/research/thesis-v2/mLearning/checkpoints/cnn_12_2025-08-02_21-07-40/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()

    # Load shape from metadata
    metadata_path = os.path.join(optimization_dir, "metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    shape_wkt = metadata["shape"]
    shape = wkt.loads(shape_wkt)

    # Get mesh size configuration for discretization
    mesh_config = get_mesh_size_presets(mesh_sizes)
    discretization_length = mesh_config['laser']

    # Sample angles
    angles, angle_indices = sample_angles_from_optimization(optimization_dir, num_samples=num_samples)
    print(f"Selected angles for simulation: {angles}")

    # Prepare output directory
    os.makedirs(output_database_dir, exist_ok=True)

    # --- Precompute mesh, grid points, and SDF field only once ---
    
    # Mesh the polygon once (use mesh for the first angle's directory, but save mesh file path for all)
    first_data_dir = os.path.join(output_database_dir, f"DATA_{0:03d}_angle_{angles[0]:.1f}")
    if not os.path.exists(first_data_dir):
        os.makedirs(first_data_dir, exist_ok=True)
    
    mesh_file = mesh_polygon(
        shape,
        mesh_sizes=mesh_sizes,
        mesh_filename=f"mesh.msh",
        output_dir=first_data_dir
    )
    mesh_stats = get_mesh_statistics_from_gmsh(mesh_file)

    # Precompute grid points and SDF field once (toolpath-independent)
    all_points = generate_grid_points(shape, discretization_length)
    grid_points = all_points[:, :2]
    grid_size = int(np.sqrt(len(grid_points)))

    # Calculate SDF field once (toolpath-independent)
    sdf_values = calculate_signed_distance_field(grid_points, shape)
    sdf_field = sdf_values.reshape((grid_size, grid_size))
    
    gmsh.initialize()
    # Now loop over angles, only recompute toolpath-dependent fields
    for i, (angle, idx) in enumerate(zip(angles, angle_indices)):
        data_dir = os.path.join(output_database_dir, f"DATA_{i:03d}_angle_{angle:.1f}")
        if os.path.exists(data_dir) and i != 0:
            print(f"Skipping angle {angle:.1f}: directory already exists")
            continue
        os.makedirs(data_dir, exist_ok=True)
        
        if i != 0:
            gmsh.model.add("thermal_3d_polygon_model")
            gmsh.merge(mesh_file)
            gmsh.write(data_dir + "/mesh.msh")
        
        # Generate toolpath for this angle
        toolpath = generate_toolpath(shape, raster_angle=angle, pattern="uniraster")
        toolpath_stats = calculate_toolpath_stats(toolpath)
        toolpath_stats['raster_angle_degrees'] = angle

        # Use preprocess_input_fields to calculate toolpath-dependent fields
        raw_fields, _, _, _ = preprocess_input_fields(shape, toolpath, all_points, sdf_field)
        sdf_field, time_field, grad_mag = raw_fields

        # Predict temperature field
        predicted_temp, _, _ = predict_temperature_field(
            model=model,
            polygon=shape,
            toolpath=toolpath,
            raster_angle=angle,
            all_points=all_points,
            sdf_field=sdf_field
        )

        fields = {
            "sdf_field": sdf_field,
            "time_field": time_field,
            "grad_mag": grad_mag,
            "predicted_temp": predicted_temp,
        }

        print(f'Saving data for angle {angle:.1f} (index {i})')
        save_data(
            output_dir=data_dir,
            shape=shape,
            toolpath=toolpath,
            toolpath_stats=toolpath_stats,
            mesh_file=mesh_file,
            fields=fields,
            mesh_stats=mesh_stats,
            inside_outside_array=all_points,
            output_name="metadata_00"
        )

    gmsh.finalize()
    print(f"Comparison DATABASE generated at: {output_database_dir}") 

# Example usage:
if __name__ == "__main__":
    # Path to optimization directory (update as needed)
    optimization_dir = "/mnt/c/Users/jamba/sim_data/OPTIMIZATION_20250803_133055"
    # Output directory for comparison database
    output_database_dir = "/mnt/c/Users/jamba/sim_data/COMPARISON_DATABASE_20250803_133055"
    generate_comparison_database(
        optimization_dir=optimization_dir,
        output_database_dir=output_database_dir,
        mesh_sizes='fine',
        num_samples=10
    )
