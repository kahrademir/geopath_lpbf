#!/usr/bin/env python3
import sys
import os
import argparse
from mpi4py import MPI
from shapely import wkt

# Add the parent directory to the Python path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver.utils import load_data, validate_polygon_geometry, save_data, mpi_print
from solver.thermal_solver_vtk import simulate

def main():
    parser = argparse.ArgumentParser(description='Run a single thermal simulation')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to the simulation data directory')
    args = parser.parse_args()
    
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    mpi_print(f"Running single simulation with {size} MPI ranks")
    mpi_print(f"Data path: {args.data_path}")
    
    try:
        # Only rank 0 loads the data, then broadcasts
        if rank == 0:
            toolpath, _, metadata, inside_outside = load_data(args.data_path, output_name="metadata_00")
            polygon = wkt.loads(metadata['shape'])
            is_valid, validation_message = validate_polygon_geometry(polygon)
            if not is_valid:
                raise ValueError(f"Polygon validation failed: {validation_message}")
            mesh_file = metadata['mesh_file']
            output_dir = args.data_path
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

        # Run simulation
        mpi_print("Starting thermal simulation...")
        max_T, domain, config_data, timing_stats = simulate(
                toolpath=toolpath,
                mesh_filename=mesh_file,
                output_dir=output_dir,
                eval_points_array=inside_outside,
                output_freq=10
            )

        # Only rank 0 saves the data
        if rank == 0:
            save_data(output_dir=output_dir,
                        fields={f"max_T": max_T},  
                        config_data=config_data,
                        timing_stats=timing_stats,
                        output_name=f"metadata_01")
            mpi_print("Simulation completed successfully")
        
    except Exception as e:
        mpi_print(f"Error in simulation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()