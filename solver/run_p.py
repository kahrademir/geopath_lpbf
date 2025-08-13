#!/usr/bin/env python3
import sys
import os
import numpy as np
import argparse
from mpi4py import MPI
from shapely import wkt

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.utils import load_data, validate_polygon_geometry, save_data, mpi_print
from solver.thermal_solver_vtk import simulate

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Only rank 0 should handle directory listing
    if rank == 0:
        data_parent_dir = os.path.join("/mnt/c/Users/jamba/sim_data/", "DATABASE_PA")
        all_dirs = [d for d in os.listdir(data_parent_dir) if os.path.isdir(os.path.join(data_parent_dir, d))]
        all_dirs.sort()
        # Filter for directories that do not contain a max_T.npy file
        all_dirs = [d for d in all_dirs if not os.path.isfile(os.path.join(data_parent_dir, d, "max_T.npy"))]
        # Filter directories based on start index
        start_index = 0
        if start_index > 0:
            all_dirs = all_dirs[start_index:]
            mpi_print(f"Starting simulation from index {start_index}, processing {len(all_dirs)} directories")
    else:
        all_dirs = None

    # Broadcast the list of directories to all ranks
    all_dirs = comm.bcast(all_dirs, root=0)

    # Print directories found and ask for confirmation (only on rank 0)
    if rank == 0:
        mpi_print(f"Found {len(all_dirs)} directories to process in {data_parent_dir}:")
        for i, dir_name in enumerate(all_dirs):
            mpi_print(f"  {i+1:3d}: {dir_name}")

        # Get user confirmation
        mpi_print("\nPress Enter to continue or 'q' to quit...")
        user_input = input().strip().lower()
        
        should_continue = user_input != 'q'
    else:
        should_continue = True

    # Broadcast the decision to all ranks
    should_continue = comm.bcast(should_continue, root=0)

    if not should_continue:
        mpi_print("Simulation cancelled by user.")
        MPI.Finalize()
        sys.exit(0)

    for data_dir_name in all_dirs:
        if rank == 0:
            data = os.path.join(data_parent_dir, data_dir_name)
            toolpath, _, metadata, inside_outside = load_data(data, output_name="metadata_00")
            polygon = wkt.loads(metadata['shape'])
            is_valid, validation_message = validate_polygon_geometry(polygon)
            if not is_valid:
                raise ValueError(f"Polygon validation failed: {validation_message}")
            mesh_file = metadata['mesh_file']
            output_dir = data
        else:
            toolpath = None
            mesh_file = None
            output_dir = None
            inside_outside = None

        # Broadcast data from rank 0 to all processes
        toolpath = comm.bcast(toolpath, root=0)
        mesh_file = comm.bcast(mesh_file, root=0)
        output_dir = comm.bcast(output_dir, root=0)
        inside_outside = comm.bcast(inside_outside, root=0)

        # Ensure all processes are synchronized before starting simulation
        comm.barrier()
        mpi_print(f"Running simulation for {data_dir_name}")
        # Run simulation with MPI - this block will use multiple cores
        max_T, domain, config_data, timing_stats = simulate(
                toolpath=toolpath,
                mesh_filename=mesh_file,
                output_dir=output_dir,
                # forced_stop_time=toolpath[-1, 0]/4,
                eval_points_array=inside_outside,
                output_freq=10,
                # forced_stop_time=toolpath[-1, 0]/4
            )

        # Only rank 0 should save the final data
        save_data(output_dir=output_dir,
                    fields={f"max_T": max_T},  
                    config_data=config_data,
                    timing_stats=timing_stats,
                    output_name=f"metadata_01")

if __name__ == "__main__":
    main() 