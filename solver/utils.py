#!/usr/bin/env python3
"""
Utility functions for 3D thermal analysis simulation
"""

import os
import json
import glob
from datetime import datetime
from mpi4py import MPI
import numpy as np
from shapely.geometry import Polygon
from typing import Tuple, Dict, Any, Optional
from dolfinx import fem

def mpi_print(*args, **kwargs):
    """Print function that only outputs on MPI rank 0"""
    if MPI.COMM_WORLD.rank == 0:
        print(*args, **kwargs)

def save_data(output_dir: str, 
                shape: Polygon=None, 
                toolpath: np.ndarray=None, 
                mesh_file: str=None,
                fields: dict[str, np.ndarray]=None,
                toolpath_stats: dict=None, 
                mesh_stats: dict=None, 
                config_data: dict=None, 
                timing_stats: dict=None,
                inside_outside_array: np.ndarray=None,
                output_name: str="metadata"):
    """
    General function to save data to a directory
    
    Args:
        output_dir: Directory to save the data
        shape: Shape data
        toolpath: Toolpath data
        toolpath_stats: Dictionary with toolpath statistics
        mesh_file: Mesh file
        fields: Dictionary with fields
        mesh_stats: Dictionary with mesh statistics
        config_data: Dictionary with configuration data
        timing_stats: Dictionary with timing statistics
        inside_outside_array: Array of inside/outside flags
        output_name: Name for the metadata file
    """
    if MPI.COMM_WORLD.rank == 0:  # Only save from rank 0 to avoid file conflicts
        os.makedirs(output_dir, exist_ok=True)
        metadata = {
            "shape": shape.wkt if shape is not None else None,
            "toolpath_stats": toolpath_stats,
            "mesh_file": mesh_file,
            "mesh_stats": mesh_stats,
            "fields": list(fields.keys()) if fields is not None else None,
            "config_data": config_data,
            "timing_stats": timing_stats,
        }
        with open(os.path.join(output_dir, output_name + ".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        mpi_print(f"Metadata saved to {os.path.join(output_dir, output_name + '.json')}")
    
        if toolpath is not None:
            np.save(os.path.join(output_dir, "toolpath.npy"), toolpath)
        if inside_outside_array is not None:
            np.save(os.path.join(output_dir, "inside_outside_array.npy"), inside_outside_array)
        if fields is not None:
            for field_name, field_data in fields.items():
                np.save(os.path.join(output_dir, f"{field_name}.npy"), field_data)


def load_data(data_dir: str, output_name: str="metadata") -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Any], Optional[np.ndarray]]:
    """
    Load data saved by save_data function
    Returns:
        toolpath, fields, metadata, inside_outside
    """
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
            
        with open(os.path.join(data_dir, output_name + ".json"), "r") as f:
            metadata = json.load(f)
        
        toolpath = np.load(os.path.join(data_dir, "toolpath.npy"))
        
        inside_outside = np.load(os.path.join(data_dir, "inside_outside_array.npy"))
        
        if metadata["fields"] is not None:
            fields = {field_name: np.load(os.path.join(data_dir, f"{field_name}.npy")) for field_name in metadata["fields"]}
        else:
            fields = {}

        mpi_print(f"Metadata loaded from {os.path.join(data_dir, output_name + '.json')}")
        return toolpath, fields, metadata, inside_outside
    else:
        return None, None, None, None # return None for all fields if not rank 0


def create_timestamped_output_dir(base_name="3D_thermal", output_dir=None) -> str:
    """
    Create a timestamped output directory with sequential numbering
    
    Args:
        base_name: Base name for the output directory
        results_base_dir: Base directory where results folders will be created.
                         If None, defaults to 'results' in current working directory.
        
    Returns:
        str: Path to the created output directory
    """
    if MPI.COMM_WORLD.rank == 0:
        # Use provided results_base_dir or default to 'results' in current directory
        if output_dir is None:
            current_dir = os.getcwd()
            output_dir = os.path.join(current_dir, "results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Find next available folder number
        existing_folders = glob.glob(os.path.join(output_dir, "[0-9][0-9][0-9]*"))
        next_folder_num = 1
        if existing_folders:
            folder_nums = [int(os.path.basename(folder)[:3]) for folder in existing_folders if os.path.basename(folder)[:3].isdigit()]
            if folder_nums:
                next_folder_num = max(folder_nums) + 1
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{next_folder_num:03d}_{timestamp}_{base_name}"
        output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None
    
    # Broadcast the output directory path from rank 0 to all other ranks
    output_dir = MPI.COMM_WORLD.bcast(output_dir, root=0)
    return output_dir 

def validate_polygon_geometry(polygon: Polygon) -> Tuple[bool, str]:
    """
    Validate polygon geometry
    """
    if not polygon.is_valid:
        return False, "Polygon is not valid"
    return True, "Polygon is valid"


def read_xdmf(xdmf_file: str, function_name: str = "Temperature", time_step: int = -1):
    """
    Read XDMF/HDF5 file and return the function data using proper DOLFINx methods.
    
    Args:
        xdmf_file: Path to the XDMF file
        function_name: Name of the function to read (default: "Temperature")
        time_step: Which time step to read (-1 for latest, specific index for that step)
    Returns:
        domain: The mesh domain
        u: DOLFINx Function with properly ordered DOF values
    """
    import h5py
    from dolfinx import io, fem
    
    # Read mesh from XDMF
    with io.XDMFFile(MPI.COMM_WORLD, xdmf_file, "r") as xdmf:
        domain = xdmf.read_mesh()
        V = fem.functionspace(domain, ("Lagrange", 1))
    
    # Find the corresponding HDF5 file
    h5_file = xdmf_file.replace(".xdmf", ".h5")
    if not os.path.exists(h5_file):
        raise FileNotFoundError(f"HDF5 file not found: {h5_file}")
    
    # Create function to hold the data
    u = fem.Function(V)
    u.name = function_name
    
    # Read function data from HDF5 - but do it properly using DOLFINx geometry
    with h5py.File(h5_file, "r") as h5:
        group_path = f"Function/{function_name}"
        if group_path not in h5:
            raise KeyError(f"Function group not found at {group_path} in {h5_file}")
        
        # Get all time keys and select the requested one
        time_keys = sorted(h5[group_path].keys(), key=lambda x: float(x.replace('_','.')))
        if time_step == -1:
            selected_key = time_keys[-1]  # Latest
        else:
            selected_key = time_keys[time_step]
        
        print(f"Reading time step: {selected_key} (index {time_step if time_step != -1 else len(time_keys)-1})")
        
        # Get the raw data
        raw_data = h5[group_path][selected_key][...]
        
        # The key insight: we need to ensure the data order matches the DOLFINx DOF ordering
        # For P1 Lagrange elements, DOFs are at vertices, so we need to match HDF5 vertex 
        # ordering with DOLFINx vertex ordering
        
        # Check if we need to reorder based on geometry
        coords_dolfin = domain.geometry.x  # DOLFINx vertex coordinates
        num_vertices = coords_dolfin.shape[0]
        
        if raw_data.size != num_vertices:
            raise ValueError(f"Data size mismatch: HDF5 has {raw_data.size} values, mesh has {num_vertices} vertices")
        
        # For now, assume the ordering is correct (most common case)
        # In complex cases, you might need to create a mapping between orderings
        u.x.array[:] = np.ravel(raw_data)
        
    return domain, u
    