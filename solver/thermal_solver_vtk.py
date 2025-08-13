#!/usr/bin/env python3
"""
Main thermal solver module for 3D thermal analysis with VTK output
"""

import numpy as np
import dolfinx
from dolfinx import mesh, fem, io
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import os
import time
from shapely.geometry import Polygon
import numpy as np

from solver.utils import mpi_print
from solver.config import *
from solver.material_models import effective_specific_heat
from solver.mesh_utils import load_mesh_from_file, eval_points, precompute_eval_points_data
from solver.heat_source import laser_heat_source_3d_toolpath

def setup_3d_thermal_problem_from_mesh(domain, dt):
    """Set up the 3D thermal problem using a pre-generated mesh with BDF2 time integration"""
    mpi_print(f"Setting up thermal problem with BDF2 time integration...")
    
    # Function spaces
    V = fem.functionspace(domain, ("Lagrange", 1))  
    
    mpi_print(f"========================\n")
    
    # Initial condition - all at ambient temperature
    T_n = fem.Function(V)  # T^n (current time step)
    T_n_minus_1 = fem.Function(V)  # T^(n-1) 
    
    # Initial conditions (ambient temperature)
    T_n.x.array[:] = AMBIENT_TEMP
    T_n_minus_1.x.array[:] = AMBIENT_TEMP
    
    # Initialize effective specific heat capacity function
    cp_eff_func = fem.Function(V)
    cp_eff_func.x.array[:] = SPECIFIC_HEAT
    
    # Create laser heat source function
    laser_source_func = fem.Function(V)
    laser_source_func.x.array[:] = 0.0
    
    # Test and trial functions
    T = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    
    # BDF2 weak form: ρcp_eff(T_n) * (3T^(n+1) - 4T^n + T^(n-1))/(2*dt) = ∇·(k∇T^(n+1)) + Q^(n+1)
    # Rearranged: (3/(2*dt)) * ρcp_eff * T^(n+1) + k∇²T^(n+1) = (4/(2*dt)) * ρcp_eff * T^n - (1/(2*dt)) * ρcp_eff * T^(n-1) + Q^(n+1)
    
    # For BDF2 scheme
    a_bdf2 = (3.0/(2.0*dt)) * DENSITY * cp_eff_func * T * v * ufl.dx + THERMAL_CONDUCTIVITY * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L_bdf2 = (4.0/(2.0*dt)) * DENSITY * cp_eff_func * T_n * v * ufl.dx - (1.0/(2.0*dt)) * DENSITY * cp_eff_func * T_n_minus_1 * v * ufl.dx + laser_source_func * v * ufl.dx
    
    # For first time step (implicit Euler when T_n_minus_1 is not available)
    a_euler = DENSITY * cp_eff_func * T * v * ufl.dx + dt * THERMAL_CONDUCTIVITY * ufl.dot(ufl.grad(T), ufl.grad(v)) * ufl.dx
    L_euler = DENSITY * cp_eff_func * T_n * v * ufl.dx + dt * laser_source_func * v * ufl.dx
    
    # Boundary conditions
    def baseplate_bottom_boundary(x):
        return np.isclose(x[2], 0.0, atol=1e-6)
    
    # Apply fixed temperature at baseplate bottom
    dofs_bottom = fem.locate_dofs_geometrical(V, baseplate_bottom_boundary)
    bc_bottom = fem.dirichletbc(fem.Constant(domain, BASEPLATE_TEMP), dofs_bottom, V)
    bcs = [bc_bottom]
    
    mpi_print(f"Boundary DOFs: {len(dofs_bottom)}")
    mpi_print(f"Using BDF2 time integration scheme (second-order accurate)")
    mpi_print(f"Using temperature-dependent specific heat capacity with latent heat of fusion")
    mpi_print(f"Method: Apparent heat capacity (C_p = C_n + C_ap)")
    mpi_print(f"Melting temperature: {MELTING_TEMP:.1f} K")
    mpi_print(f"Latent heat of fusion: {LATENT_HEAT:.0f} J/kg")
    mpi_print(f"Temperature range: {MELTING_TEMP - MELTING_RANGE/2:.1f} - {MELTING_TEMP + MELTING_RANGE/2:.1f} K")
    
    return domain, V, T_n, T_n_minus_1, laser_source_func, cp_eff_func, a_bdf2, L_bdf2, a_euler, L_euler, bcs

def create_vtk_files(output_dir, domain):
    """Create VTK files for output with proper MPI coordination"""
    vtk_files = {}
    
    # Create separate subdirectories for each VTK file type
    temperature_dir = os.path.join(output_dir, "temperature")
    cp_effective_dir = os.path.join(output_dir, "cp_effective")
    max_temperature_dir = os.path.join(output_dir, "max_temperature")
    
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(temperature_dir, exist_ok=True)
        os.makedirs(cp_effective_dir, exist_ok=True)
        os.makedirs(max_temperature_dir, exist_ok=True)
    
    # Ensure all processes wait before creating files
    MPI.COMM_WORLD.barrier()
    
    # Temperature file
    vtk_file = os.path.join(temperature_dir, "temperature_3d_simulation.vtk")
    try:
        # Create VTK file with proper MPI coordination
        vtk = io.VTKFile(MPI.COMM_WORLD, vtk_file, "w")
        vtk.write_mesh(domain)
        vtk_files['temperature'] = vtk
        mpi_print(f"Successfully created temperature VTK file: {vtk_file}")
    except Exception as e:
        mpi_print(f"Error creating VTK temperature file: {e}")
        mpi_print(f"This may be due to MPI coordination issues or file permissions")
        # Don't set to None - let it fail properly to identify the real issue
        raise
    
    # Effective specific heat file
    vtk_cp_file = os.path.join(cp_effective_dir, "cp_effective_3d_simulation.vtk")
    try:
        vtk_cp = io.VTKFile(MPI.COMM_WORLD, vtk_cp_file, "w")
        vtk_cp.write_mesh(domain)
        vtk_files['cp_effective'] = vtk_cp
        mpi_print(f"Successfully created cp_effective VTK file: {vtk_cp_file}")
    except Exception as e:
        mpi_print(f"Error creating VTK cp file: {e}")
        mpi_print(f"This may be due to MPI coordination issues or file permissions")
        raise
    
    # Maximum temperature file
    vtk_max_file = os.path.join(max_temperature_dir, "max_temperature_3d_simulation.vtk")
    try:
        vtk_max = io.VTKFile(MPI.COMM_WORLD, vtk_max_file, "w")
        vtk_max.write_mesh(domain)
        vtk_files['max_temperature'] = vtk_max
        mpi_print(f"Successfully created max_temperature VTK file: {vtk_max_file}")
    except Exception as e:
        mpi_print(f"Error creating VTK max temperature file: {e}")
        mpi_print(f"This may be due to MPI coordination issues or file permissions")
        raise
    
    # Ensure all processes have created files before proceeding
    MPI.COMM_WORLD.barrier()
    
    return vtk_files

def simulate(
        mesh_filename: str,
        toolpath: np.ndarray,
        dt=DEFAULT_DT,
        output_dir=None,
        forced_stop_time=None,
        eval_points_array=None,
        output_freq = 10
        ):
    """
    Run the complete 3D thermal simulation with pre-generated mesh and toolpath
    
    Args:
        mesh_filename: Path to pre-generated mesh file (required)
        toolpath: Pre-generated toolpath array (required)
        dt: Time step for simulation
        output_dir: Directory to save results (optional, defaults to mesh_filename directory)
        forced_stop_time: Optional time to stop simulation early
        eval_points_array: Optional numpy array of points [(x, y), ...] or [(x, y, z), ...] to evaluate temperature at
    
    Returns:
        reduced_points_max: Maximum temperature field at the evaluated points (toolpath or evaluation points)
        domain: The mesh domain
        config_data: Simulation configuration data
        timing_stats: Timing statistics
    """

    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(mesh_filename)
        mpi_print(f"Using output directory from mesh filename: {output_dir}")
    else:
        mpi_print(f"Using provided output directory: {output_dir}")
    
    # Start timing the simulation
    simulation_start_time = time.time()
    wall_clock_start = time.perf_counter()
    cpu_start_time = time.process_time()
    
    mpi_print("Starting 3D thermal simulation...")
    mpi_print(f"\n=== SIMULATION TIMING STARTED ===")
    mpi_print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(simulation_start_time))}")
    mpi_print(f"==================================\n")
    
    # Load pre-generated mesh
    mpi_print(f"Loading pre-generated mesh: {mesh_filename}")
    domain, _, _ = load_mesh_from_file(mesh_filename)
    
    # Use pre-generated toolpath
    mpi_print(f"Using pre-generated toolpath:")
    mpi_print(f"  - Points: {len(toolpath)}")
    mpi_print(f"  - Total time: {toolpath[-1,0]:.3f}s")
    
    # Set up thermal problem
    domain, V, T_n, T_n_minus_1, laser_source_func, cp_eff_func, a_bdf2, L_bdf2, a_euler, L_euler, bcs = setup_3d_thermal_problem_from_mesh(domain, dt)
    
    # Initialize maximum temperature tracking
    T_max = fem.Function(V)  # Maximum temperature experienced at each point
    T_max.x.array[:] = AMBIENT_TEMP  # Initialize with ambient temperature   
    
    # Save simulation metadata and toolpath data
    config_data = {
        "geometry": {
            "part_height_m": PART_HEIGHT,
            "baseplate_width_m": BASEPLATE_WIDTH,
            "baseplate_length_m": BASEPLATE_LENGTH,
            "baseplate_thickness_m": BASEPLATE_THICKNESS,
            "lasered_layer_depth_m": LASERED_LAYER_DEPTH
        },
        "laser_parameters": {
            "power_W": LASER_POWER,
            "diameter_m": LASER_DIAMETER,
            "radius_m": LASER_RADIUS,
            "scan_speed_m_per_s": SCAN_SPEED,
            "absorption_coefficient": ABSORPTION_COEFF,
        },
        "material_properties": {
            "density_kg_per_m3": DENSITY,
            "specific_heat_J_per_kg_K": SPECIFIC_HEAT,
            "thermal_conductivity_W_per_m_K": THERMAL_CONDUCTIVITY,
            "melting_temperature_K": MELTING_TEMP,
            "latent_heat_fusion_J_per_kg": LATENT_HEAT,
            "melting_range_K": MELTING_RANGE
        },
        "boundary_conditions": {
            "ambient_temperature_K": AMBIENT_TEMP,
            "baseplate_temperature_K": BASEPLATE_TEMP
        }
    }
    
    # Create VTK files
    vtk_files = create_vtk_files(output_dir, domain)
    
    # Create subdirectory for temperature point data
    temp_points_output_dir = os.path.join(output_dir, "T_points")
    if MPI.COMM_WORLD.rank == 0:
        os.makedirs(temp_points_output_dir, exist_ok=True)
    MPI.COMM_WORLD.barrier()
    
    # Simulation parameters  
    laser_time = toolpath[-1,0]
    cooling_time = COOLING_TIME
    final_time = laser_time + cooling_time
    cooling_dt = dt * COOLING_DT_MULTIPLIER  # Larger timestep for cooling
    
    # Calculate approximate total steps
    laser_steps = int(laser_time / dt)
    cooling_steps = int(cooling_time / cooling_dt)
    total_steps = laser_steps + cooling_steps
    
    mpi_print(f"Running simulation with adaptive timestep:")
    mpi_print(f"  - Laser phase: ~{laser_steps} steps, dt={dt:.4f}s")
    mpi_print(f"  - Cooling phase: ~{cooling_steps} steps, dt={cooling_dt:.4f}s")
    mpi_print(f"  - Total time: {final_time:.3f}s")
    mpi_print(f"Results will be saved to separate subdirectories in: {output_dir}")
    mpi_print(f"  - Temperature: {os.path.join(output_dir, 'temperature')}")
    mpi_print(f"  - Effective specific heat: {os.path.join(output_dir, 'cp_effective')}")
    mpi_print(f"  - Maximum temperature: {os.path.join(output_dir, 'max_temperature')}")
    mpi_print(f"  - Temperature point data: {os.path.join(output_dir, 'T_points')}")
    
    # Determine which points to evaluate on
    if eval_points_array is not None:
        # Only evaluate on points inside the shape (eval_points_array is a numpy array of [x, y, is_inside] where is_inside is 1 for inside and 0 for outside)
        size = int(np.sqrt(len(eval_points_array)))
        inside_mask = eval_points_array[:, 2] == 1  
        points_to_evaluate = eval_points_array[inside_mask][:, :2]
    else:
        points_to_evaluate = toolpath[:, 1:3]  # Use toolpath points if no evaluation array provided
    
    # Precompute eval_points data for the points we'll evaluate on
    mpi_print(f"Precomputing cell collision data for evaluation points...")
    z_coordinate = BASEPLATE_THICKNESS+PART_HEIGHT-1e-6
    eval_points_precomputed = precompute_eval_points_data(
        domain, points_to_evaluate, z_coordinate=z_coordinate
    )
    found_points = len(eval_points_precomputed[0])
    mpi_print(f"Precomputed data for {len(points_to_evaluate)} points, found {found_points} points on this process")
    
    # Time loop
    current_time = 0.0
    timestep = 0
    current_dt = dt  # Initialize current timestep
    
    # For BDF2, we need to track if this is the first step
    is_first_step = True
    problem = None  # Initialize problem
    
    # Timing tracking for laser phase
    laser_timestep_times = []  # Store wall clock times for laser phase timesteps
    laser_timestep_count = 0   # Count of completed laser phase timesteps
    
    # Initialize tracking array for the points we're evaluating
    points_max = np.zeros(len(points_to_evaluate))
    
    while current_time < final_time and (forced_stop_time is None or current_time < forced_stop_time):
        # Use larger timestep during cooling phase
        new_dt = cooling_dt if current_time > laser_time else dt
        phase = "cooling" if current_time > laser_time else "laser"
        
        # Start timing this timestep if in laser phase
        timestep_start_time = time.perf_counter() if phase == "laser" else None
        
        current_time += current_dt
        timestep += 1
        
        # Update effective specific heat capacity based on current temperature
        cp_eff_values = effective_specific_heat(T_n.x.array)
        cp_eff_func.x.array[:] = cp_eff_values
        
        # Update laser heat source based on current time and toolpath
        if phase == "cooling":
            # During cooling phase, laser is off - set to zero
            laser_source_func.x.array[:] = 0.0
            output_freq = 10  # For checking when laser turns off
            if timestep % output_freq == 0 and timestep == int(laser_time / dt) + 1:
                mpi_print(f"Laser turned OFF - cooling phase started")
        else:
            # During laser phase, use toolpath
            laser_values = laser_heat_source_3d_toolpath(domain.geometry.x.T, current_time, toolpath, LASER_POWER, LASER_RADIUS, ABSORPTION_COEFF)
            laser_source_func.x.array[:] = laser_values
        
        current_dt = new_dt
        
        if is_first_step:
            # Use implicit Euler for first step
            a_euler_current = DENSITY * cp_eff_func * ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx + current_dt * THERMAL_CONDUCTIVITY * ufl.dot(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx
            L_euler_current = DENSITY * cp_eff_func * T_n * ufl.TestFunction(V) * ufl.dx + current_dt * laser_source_func * ufl.TestFunction(V) * ufl.dx
            problem = LinearProblem(a_euler_current, L_euler_current, bcs=bcs, petsc_options=PETSC_OPTIONS)
            scheme_name = "Euler"
            if timestep == 1:
                mpi_print(f"First timestep using implicit Euler scheme")
            is_first_step = False
        else:
            # Use BDF2 for subsequent steps
            # Always recreate BDF2 problem since it depends on updated functions
            a_bdf2_current = (3.0/(2.0*current_dt)) * DENSITY * cp_eff_func * ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx + THERMAL_CONDUCTIVITY * ufl.dot(ufl.grad(ufl.TrialFunction(V)), ufl.grad(ufl.TestFunction(V))) * ufl.dx
            L_bdf2_current = (4.0/(2.0*current_dt)) * DENSITY * cp_eff_func * T_n * ufl.TestFunction(V) * ufl.dx - (1.0/(2.0*current_dt)) * DENSITY * cp_eff_func * T_n_minus_1 * ufl.TestFunction(V) * ufl.dx + laser_source_func * ufl.TestFunction(V) * ufl.dx
            problem = LinearProblem(a_bdf2_current, L_bdf2_current, bcs=bcs, petsc_options=PETSC_OPTIONS)
            scheme_name = "BDF2"
            if timestep == 2:
                mpi_print(f"Switched to BDF2 scheme (second-order accurate)")
        
        # Solve for current timestep
        T_new = problem.solve()
        

        # Evaluate temperature at the specified points
        T_points = eval_points(
            domain, points_to_evaluate, T_new, precomputed_data=eval_points_precomputed
        )

        # Gather values from all ranks to rank 0
        gathered_values = MPI.COMM_WORLD.gather(T_points, root=0)

        if MPI.COMM_WORLD.rank == 0:
            gathered_values = np.array(gathered_values).T
            combined_values = np.zeros(gathered_values.shape[0])
            # Vectorized approach to find first non-NaN value in each row
            non_nan_mask = ~np.isnan(gathered_values)
            first_non_nan_indices = np.argmax(non_nan_mask, axis=1)
            # Handle rows where all values are NaN
            combined_values = gathered_values[np.arange(gathered_values.shape[0]), first_non_nan_indices]

            # Save as .npy in the original eval_points_array shape (if provided)
            # if eval_points_array is not None:
            #     T_points_full = np.zeros(len(eval_points_array), dtype=np.float64)
            #     # Only save the values for the points inside the shape
            #     T_points_full[inside_mask] = combined_values
            #     T_points_full = T_points_full.reshape((size, size))
                # np.save(os.path.join(temp_points_output_dir, f"T_points_{timestep:05d}.npy"), T_points_full)
            # else:
                # np.save(os.path.join(temp_points_output_dir, f"T_points_{timestep:05d}.npy"), combined_values)

        
        # Track timing for laser phase timesteps
        if phase == "laser" and timestep_start_time is not None:
            timestep_end_time = time.perf_counter()
            timestep_wall_time = timestep_end_time - timestep_start_time
            laser_timestep_times.append(timestep_wall_time)
            laser_timestep_count += 1
            
            # Keep only the last 50 timesteps for a rolling average
            if len(laser_timestep_times) > 50:
                laser_timestep_times = laser_timestep_times[-50:]
        
        # Update maximum temperature tracking
        # For each point, keep the maximum between current max and new temperature
        T_max.x.array[:] = np.maximum(T_max.x.array, T_new.x.array)
        
        # Update maximum temperature tracking for the evaluated points
        gathered_values = MPI.COMM_WORLD.gather(T_points, root=0)
        
        if MPI.COMM_WORLD.rank == 0:
            gathered_values = np.array(gathered_values).T
            combined_values = np.zeros(gathered_values.shape[0])
            # Vectorized approach to find first non-NaN value in each row
            non_nan_mask = ~np.isnan(gathered_values)
            first_non_nan_indices = np.argmax(non_nan_mask, axis=1)
            # Handle rows where all values are NaN
            combined_values = gathered_values[np.arange(gathered_values.shape[0]), first_non_nan_indices]
            
            # Update maximum temperatures
            points_max = np.maximum(points_max, combined_values)
        else:
            points_max = None
        
        # Update for next timestep - store current solution as T_n_minus_1, new solution as T_n
        T_n_minus_1.x.array[:] = T_n.x.array[:]  # T^(n-1) = T^n
        T_n.x.array[:] = T_new.x.array[:]        # T^n = T^(n+1)
        
        # Output progress and write files
        
         # Reduced I/O frequency
        if timestep % output_freq == 0:
            max_temp = np.max(T_new.x.array)
            max_temp_overall = np.max(T_max.x.array)
            max_cp_eff = np.max(cp_eff_func.x.array)

            # Only print from rank 0 and flush immediately
            if MPI.COMM_WORLD.rank == 0:
                # Calculate remaining time estimate for laser phase
                remaining_time_str = ""
                if phase == "laser" and len(laser_timestep_times) >= 5:  # Need at least 5 samples for reasonable estimate
                    avg_timestep_wall_time = np.mean(laser_timestep_times)
                    remaining_sim_time = laser_time - current_time
                    remaining_sim_timesteps = remaining_sim_time / dt
                    estimated_remaining_wall_time = remaining_sim_timesteps * avg_timestep_wall_time
                    
                    # Format the remaining time nicely
                    if estimated_remaining_wall_time > 3600:  # More than 1 hour
                        hours = int(estimated_remaining_wall_time // 3600)
                        minutes = int((estimated_remaining_wall_time % 3600) // 60)
                        remaining_time_str = f", Est. remaining: {hours}h {minutes}m"
                    elif estimated_remaining_wall_time > 60:  # More than 1 minute
                        minutes = int(estimated_remaining_wall_time // 60)
                        seconds = int(estimated_remaining_wall_time % 60)
                        remaining_time_str = f", Est. remaining: {minutes}m {seconds}s"
                    else:  # Less than 1 minute
                        remaining_time_str = f", Est. remaining: {estimated_remaining_wall_time:.0f}s"
                print(f"Time: {current_time:.3f}s, Max temp: {max_temp:.1f}K, Overall max: {max_temp_overall:.1f}K, Max cp_eff: {max_cp_eff:.0f}J/kg·K ({phase} phase, {scheme_name}), Elapsed time: {time.perf_counter() - wall_clock_start:0.0f}s {remaining_time_str}", flush=True)

            # Write VTK files with MPI coordination
            try:
                # T_new.name = "Temperature"
                # vtk_files['temperature'].write_function(T_new, current_time)
                # # if timestep % (output_freq*1) == 0:
                # cp_eff_func.name = "Effective_Specific_Heat"
                # vtk_files['cp_effective'].write_function(cp_eff_func, current_time)
                # T_max.name = "Maximum_Temperature"
                # vtk_files['max_temperature'].write_function(T_max, current_time)           
                # # Ensure all processes complete writing before continuing
                MPI.COMM_WORLD.barrier()
            except Exception as e:
                mpi_print(f"Error writing to VTK files at time {current_time}: {e}")
                # Continue simulation even if file writing fails
                pass
    
    # Write final maximum temperature with MPI coordination
    try:
        T_max.name = "Maximum_Temperature"
        vtk_files['max_temperature'].write_function(T_max, final_time)
        # Ensure all processes complete final writing
        MPI.COMM_WORLD.barrier()
        mpi_print("Final VTK writes completed successfully")
    except Exception as e:
        mpi_print(f"Error writing final VTK data: {e}")
    
    # Close the VTK files with proper MPI coordination
    mpi_print("Closing VTK files...")
    try:
        for key, vtk in vtk_files.items():
            vtk.close()
        # Ensure all processes complete file closing before proceeding
        MPI.COMM_WORLD.barrier()
        mpi_print("All VTK files closed successfully")
    except Exception as e:
        mpi_print(f"Error closing VTK files: {e}")
        # Try to close individual files if collective close fails
        for key, vtk in vtk_files.items():
            try:
                vtk.close()
            except:
                mpi_print(f"Failed to close {key} VTK file")
    
    # final_max_temp = np.max(T_max.x.array)
    # final_current_temp = np.max(T_new.x.array)
    
    # Calculate timing statistics
    simulation_end_time = time.time()
    wall_clock_end = time.perf_counter()
    cpu_end_time = time.process_time()
    
    total_wall_time = wall_clock_end - wall_clock_start
    total_cpu_time = cpu_end_time - cpu_start_time
    total_simulation_time = simulation_end_time - simulation_start_time
    
    # Calculate performance metrics
    timesteps_per_second = timestep / total_wall_time if total_wall_time > 0 else 0
    dofs_timesteps_per_second = (V.dofmap.index_map.size_local * timestep) / total_wall_time if total_wall_time > 0 else 0
    
    # Create timing statistics dictionary
    timing_stats = {
        "timestep_laser_phase": dt,
        "timestep_cooling_phase": dt * 100,
        "start_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(simulation_start_time)),
        "end_time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(simulation_end_time)),
        "total_wall_clock_time_seconds": total_wall_time,
        "total_cpu_time_seconds": total_cpu_time,
        "cpu_efficiency_percent": (total_cpu_time/total_wall_time)*100 if total_wall_time > 0 else 0,
        "timesteps_per_second": timesteps_per_second,
        "dof_timesteps_per_second": dofs_timesteps_per_second,
        "average_time_per_timestep_seconds": total_wall_time/timestep if timestep > 0 else 0,
        "total_timesteps": timestep
    }
    
    # Ensure all MPI ranks have the same data before saving
    # Use MPI to gather and reduce the data to ensure consistency
    
    # Gather all points_max arrays from all ranks
    # Ensure all ranks send a valid array (never None)
    # if points_max is None:
    #     points_max_to_send = np.zeros(len(points_to_evaluate))
    # else:
    #     points_max_to_send = points_max

    # gathered_points_max = MPI.COMM_WORLD.gather(points_max_to_send, root=0)

    # if MPI.COMM_WORLD.rank == 0:
    #     reduced_points_max = np.maximum.reduce(gathered_points_max)
    #     # Save the final temperatures (replace CSV with .npy, MPI safe)
    #     if eval_points_array is not None:
    #         T_points_full = np.zeros(len(eval_points_array), dtype=np.float64)
    #         T_points_full[inside_mask] = reduced_points_max
    #         T_points_full = T_points_full.reshape((size, size))
    #         np.save(os.path.join(temp_points_output_dir, "T_points_final.npy"), T_points_full)
    #         reduced_points_max = T_points_full
    #     else:
    #         np.save(os.path.join(temp_points_output_dir, "T_points_final.npy"), reduced_points_max)
    # else:
    #     reduced_points_max = np.zeros_like(points_max_to_send)

    # # Broadcast the final result back to all ranks for return
    # reduced_points_max = MPI.COMM_WORLD.bcast(reduced_points_max, root=0)

    mpi_print("\nSimulation completed successfully")
    

    mpi_print(f"Mesh file: {mesh_filename}")
    mpi_print(f"Total timesteps: {timestep} (BDF2 with adaptive timestep: laser≈{laser_steps}, cooling≈{timestep-laser_steps})")

    # return reduced_points_max, domain, config_data, timing_stats
    return None, domain, config_data, timing_stats