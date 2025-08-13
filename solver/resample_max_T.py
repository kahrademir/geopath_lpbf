# %%
#!/usr/bin/env python3
"""
Resample maximum temperature values from VTK files at arbitrary point coordinates
"""

import numpy as np
import pyvista as pv
import os
from scipy.interpolate import griddata, RBFInterpolator
import sys
sys.path.append('/home/jamba/research/thesis-v2')

from solver.config import BASEPLATE_THICKNESS, PART_HEIGHT, get_mesh_size_presets  
from solver.utils import load_data
from geometry.geometry_samples import generate_grid_points
from shapely.wkt import loads
from shapely.geometry import Polygon
 
def sample_max_temperature_from_vtk(vtk_file_path, sample_points, z_coordinate=BASEPLATE_THICKNESS + PART_HEIGHT - 1e-6):
    """
    Sample maximum temperature at specified points from VTK files
    
    Args:
        vtk_file_path: Path to .pvtu file (handles parallel structure automatically)
        sample_points: numpy array of coordinates
                      - Shape (N, 2) for 2D points [x, y] - requires z_coordinate parameter
                      - Shape (N, 3) for 3D points [x, y, z] - z_coordinate ignored
        z_coordinate: float, z-level to sample at when sample_points is 2D (optional)
    
    Returns:
        temperatures: numpy array of interpolated maximum temperatures at sample points
    """
    
    # Load the parallel VTK dataset (automatically handles .pvtu + .vtu structure)
    print(f"Loading VTK file: {vtk_file_path}")
    mesh = pv.read(vtk_file_path)
    
    print(f"Loaded mesh with {mesh.n_points} points and {mesh.n_cells} cells")
    print(f"Available arrays: {list(mesh.array_names)}")
    
    # Handle 2D vs 3D coordinates
    if sample_points.shape[1] == 2:
        if z_coordinate is None:
            raise ValueError("z_coordinate must be provided when sample_points is 2D")
        # Convert 2D to 3D by adding z-coordinate
        points_3d = np.column_stack([sample_points, np.full(len(sample_points), z_coordinate)])
        print(f"Converting {len(sample_points)} 2D points to 3D using z = {z_coordinate}")
    elif sample_points.shape[1] == 3:
        points_3d = sample_points
        print(f"Using {len(sample_points)} 3D points directly")
    else:
        raise ValueError(f"sample_points must have shape (N, 2) or (N, 3), got {sample_points.shape}")
    
    print(f"Sampling at {len(points_3d)} points...")
    print(f"Point coordinate ranges:")
    print(f"  X: [{points_3d[:, 0].min():.6f}, {points_3d[:, 0].max():.6f}]")
    print(f"  Y: [{points_3d[:, 1].min():.6f}, {points_3d[:, 1].max():.6f}]")
    print(f"  Z: [{points_3d[:, 2].min():.6f}, {points_3d[:, 2].max():.6f}]")
    
    # Sample using PyVista's interpolation
    # This creates a new mesh with points at the sample locations
    point_cloud = pv.PolyData(points_3d)

    sampled_mesh = point_cloud.sample(mesh)
    
    # Extract the maximum temperature field
    # The field name should match what was written in thermal_solver_vtk.py
    temperature_field_name = "Maximum_Temperature"
    
    if temperature_field_name not in sampled_mesh.array_names:
        available_names = list(sampled_mesh.array_names)
        raise ValueError(f"Temperature field '{temperature_field_name}' not found. Available: {available_names}")
    
    temperatures = sampled_mesh[temperature_field_name]
    
    print(f"Successfully sampled {len(temperatures)} temperature values")
    print(f"Temperature range: [{temperatures.min():.1f}, {temperatures.max():.1f}] K")
    
    return temperatures

def get_latest_max_temperature_file(simulation_dir):
    """
    Find the latest (highest timestep) .pvtu file in the max_temperature directory
    
    Args:
        simulation_dir: Path to simulation directory (contains max_temperature subdirectory)
    
    Returns:
        Path to the latest .pvtu file
    """
    max_temp_dir = os.path.join(simulation_dir, "max_temperature")
    
    if not os.path.exists(max_temp_dir):
        raise FileNotFoundError(f"Max temperature directory not found: {max_temp_dir}")
    
    # Find all .pvtu files
    pvtu_files = [f for f in os.listdir(max_temp_dir) if f.endswith('.pvtu')]
    
    if not pvtu_files:
        raise FileNotFoundError(f"No .pvtu files found in {max_temp_dir}")
    
    # Sort by timestep number (extract number from filename)
    def extract_timestep(filename):
        # Files are named like: max_temperature_3d_simulation000001.pvtu
        import re
        match = re.search(r'(\d+)\.pvtu$', filename)
        return int(match.group(1)) if match else -1
    
    latest_file = max(pvtu_files, key=extract_timestep)
    latest_path = os.path.join(max_temp_dir, latest_file)
    
    print(f"Found latest max temperature file: {latest_file}")
    
    return latest_path

def process_single_directory(simulation_dir, output_filename="max_T_resampled.npy"):
    """
    Process a single DATA directory: resample VTK max temperatures and save to specified file
    
    Args:
        simulation_dir: Path to DATA_XXX directory
        output_filename: Name of output file (e.g., "max_T.npy", "max_T_resampled.npy")
    
    Returns:
        bool: True if successful, False if failed
    """
    try:
        print(f"\n=== Processing {os.path.basename(simulation_dir)} ===")
        
        # Check if max_temperature directory exists
        max_temp_dir = os.path.join(simulation_dir, "max_temperature")
        if not os.path.exists(max_temp_dir):
            print(f"❌ No max_temperature directory found in {simulation_dir}")
            return False
        
        # Load toolpath and metadata
        print("📂 Loading toolpath and metadata...")
        toolpath, _, metadata, inside_outside = load_data(simulation_dir, output_name="metadata_00")
        
        # Extract toolpath points for temperature sampling
        if len(toolpath) == 0:
            print("❌ No toolpath found, skipping...")
            return False
        
        test_points_2d = toolpath[:, 1:3]  # Extract x,y coordinates (skip time)
        print(f"📍 Using {len(test_points_2d)} toolpath points for temperature sampling")
        
        # Get the latest max temperature VTK file
        print("🔍 Finding latest VTK file...")
        vtk_file = get_latest_max_temperature_file(simulation_dir)
        
        # Sample temperatures at toolpath points
        print("🌡️  Sampling temperatures from VTK...")
        z_level = BASEPLATE_THICKNESS + PART_HEIGHT - 1e-6
        temperatures_2d = sample_max_temperature_from_vtk(vtk_file, test_points_2d, z_coordinate=z_level)
        
        # Use the same grid points that were used for other fields
        print("🔲 Using saved grid points from inside_outside_array...")
        
        # Use the already-saved grid points instead of regenerating them
        # This ensures consistency with sdf_field, time_field, grad_mag
        all_points = inside_outside
        grid_points = all_points[:, :2]
        grid_size = int(np.sqrt(len(grid_points)))
        
        print(f"📐 Generated {len(grid_points)} grid points ({grid_size}x{grid_size})")
        
        # Interpolate temperatures from toolpath points to grid points
        print("🔄 Interpolating temperatures to grid...")
        
        # Use RBF interpolation for smoother temperature field

        rbf_interp = RBFInterpolator(test_points_2d, temperatures_2d, kernel='thin_plate_spline')
        temperature_grid = rbf_interp(grid_points)

        
        # Apply shape mask (only keep temperatures inside the shape)
        temperature_grid = np.where(all_points[:, 2] == 1, temperature_grid, 0)
        temperature_grid = temperature_grid.reshape(grid_size, grid_size)
        
        # Save the resampled max temperature grid
        output_file = os.path.join(simulation_dir, output_filename)
        
        # Check if file exists and warn about overwrite
        if os.path.exists(output_file):
            print(f"⚠️  File already exists: {output_filename} - will overwrite")
        
        print(f"💾 Saving resampled temperatures to {output_file}")
        np.save(output_file, temperature_grid)
        
        # Print statistics
        valid_temps = temperature_grid[~np.isnan(temperature_grid)]
        if len(valid_temps) > 0:
            print(f"✅ Successfully processed {os.path.basename(simulation_dir)}")
            print(f"   Temperature range: [{valid_temps.min():.1f}, {valid_temps.max():.1f}] K")
            print(f"   Valid points: {len(valid_temps)}/{len(grid_points)}")
        else:
            print(f"⚠️  Warning: No valid temperatures found in {os.path.basename(simulation_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {os.path.basename(simulation_dir)}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def process_all_database_directories(database_path, output_filename="max_T_resampled.npy"):
    """
    Process all DATA_XXX directories in the database
    
    Args:
        database_path: Path to DATABASE_101 directory
        output_filename: Name of output file for each directory (e.g., "max_T.npy", "max_T_resampled.npy")
    """
    print(f"🚀 Starting batch processing of DATABASE_101")
    print(f"Database path: {database_path}")
    print(f"Output filename: {output_filename}")
    
    # Find all DATA_XXX directories
    if not os.path.exists(database_path):
        raise FileNotFoundError(f"Database directory not found: {database_path}")
    
    data_dirs = []
    for item in os.listdir(database_path):
        item_path = os.path.join(database_path, item)
        if os.path.isdir(item_path) and item.startswith('DATA_'):
            data_dirs.append(item_path)
    
    data_dirs.sort()  # Process in order
    print(f"📁 Found {len(data_dirs)} DATA directories to process")
    
    if len(data_dirs) == 0:
        print("❌ No DATA_XXX directories found!")
        return
    
    # Process each directory
    successful = 0
    failed = 0
    
    for i, data_dir in enumerate(data_dirs, 1):
        print(f"\n⏳ Progress: {i}/{len(data_dirs)} directories")
        
        if process_single_directory(data_dir, output_filename):
            successful += 1
        else:
            failed += 1
    
    # Final summary
    print(f"\n🎯 BATCH PROCESSING COMPLETE")
    print(f"✅ Successfully processed: {successful} directories")
    print(f"❌ Failed: {failed} directories")
    print(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")

# %%
if __name__ == "__main__":
    # Configuration
    DATABASE_PATH = "/mnt/c/Users/jamba/sim_data/DATABASE_PATTERNS"
    
    print(f"Target database: {DATABASE_PATH}")
    print("\nUsage examples:")
    print("1. Create new files (safe):   process_all_database_directories(DATABASE_PATH, 'max_T_resampled.npy')")
    print("2. Overwrite existing files:  process_all_database_directories(DATABASE_PATH, 'max_T.npy')")
    print("3. Test single directory:     process_single_directory('/path/to/DATA_000', 'test_output.npy')")
    
    # Choose your option:
    
    # Option 1: Create new resampled files (safe - doesn't overwrite existing max_T.npy)
    process_all_database_directories(DATABASE_PATH, "max_T_tp.npy")
    
    # Option 2: Overwrite existing max_T.npy files (DESTRUCTIVE!)
    # process_all_database_directories(DATABASE_PATH, "max_T.npy")
    
    # Option 3: Test on single directory first
    # process_single_directory("/mnt/c/Users/jamba/sim_data/DATABASE_101/DATA_000", "max_T_tp.npy")

# %%
