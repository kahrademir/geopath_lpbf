#!/usr/bin/env python3

"""
3D Thermal Mesh Generator using Gmsh
=====================================

Generates a 3D mesh for thermal analysis of laser powder bed fusion.
Domain consists of baseplate, part, and lasered layer regions.
"""

import numpy as np
import os
import subprocess
import sys
import traceback
from mpi4py import MPI
from shapely.geometry import Polygon, Point
import gmsh
from typing import Tuple, Dict

from solver.config import get_mesh_size_presets, BASEPLATE_WIDTH, BASEPLATE_LENGTH, BASEPLATE_THICKNESS, LASER_DIAMETER
from solver.config import PART_HEIGHT
from solver.utils import mpi_print

def launch_gmsh_gui(mesh_file):
    """Launch Gmsh GUI with the generated mesh file"""
    
    if not os.path.exists(mesh_file):
        mpi_print(f"Mesh file {mesh_file} not found.")
        return
    
    try:
        import gmsh
        mpi_print(f"Launching Gmsh GUI with mesh file: {mesh_file}")
        
        gmsh.initialize()
        gmsh.open(mesh_file)
        
        # Set visualization options
        gmsh.option.setNumber("Mesh.VolumeEdges", 1)
        gmsh.option.setNumber("Mesh.VolumeFaces", 1)
        
        if "-nopopup" not in sys.argv:
            gmsh.fltk.run()
        
        gmsh.finalize()
        
    except ImportError:
        mpi_print("Gmsh not available for GUI.")
        try:
            subprocess.run(["gmsh", mesh_file], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            mpi_print(f"You can manually open {mesh_file} with Gmsh to visualize the mesh.")
    
    except Exception as e:
        mpi_print(f"Error launching Gmsh GUI: {e}")

def mesh_polygon(polygon: Polygon, 
                                baseplate_dims: Tuple[float, float, float]=(BASEPLATE_WIDTH, BASEPLATE_LENGTH, BASEPLATE_THICKNESS),
                                part_height: float=PART_HEIGHT, 
                                lasered_layer_depth: float=LASER_DIAMETER/2, 
                                mesh_filename: str = "polygon_thermal_mesh_3d.msh", 
                                mesh_sizes: Dict[str, float] = 'coarse', 
                                output_dir: str = None) -> str:
    """
    Generate a 3D thermal mesh using Gmsh API with polygon-shaped part and lasered layer
    
    Args:
        polygon: Polygon object defining the shape of the part and lasered layer
        baseplate_dims: (width, length, thickness) of the baseplate in meters  
        part_height: Total height of the part in meters
        lasered_layer_depth: Depth of the lasered layer in meters
        output_file: Output mesh filename
        mesh_sizes: Dictionary with mesh sizes for different domains:
                   {'laser': size, 'part': size, 'baseplate': size}
                   If None, uses default values based on laser_diameter
        output_dir: Directory to save the mesh file (if None, uses default meshes directory)
        
    Returns:
        output_file: Path to the generated mesh file
    """
    
    try:

        mpi_print("Using Gmsh to generate 3D thermal mesh with polygon shape...")
        
        # Input validation
        if not isinstance(polygon, Polygon):
            raise ValueError("polygon must be a Shapely Polygon object")
        if len(polygon.exterior.xy[0]) < 4:  # Shapely coords include closing point, so 3 vertices = 4 coords
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Determine output directory
        if output_dir is not None:
            meshes_dir = output_dir
            os.makedirs(meshes_dir, exist_ok=True)
            output_file = os.path.join(meshes_dir, mesh_filename)
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            meshes_dir = os.path.join(script_dir, "meshes")
            os.makedirs(meshes_dir, exist_ok=True)
            output_file = os.path.join(meshes_dir, output_file)
        
        # Initialize Gmsh
        gmsh.initialize()
        gmsh.model.add("thermal_3d_polygon_model")
        
        mesh_sizes = get_mesh_size_presets(mesh_sizes)
        # Set mesh size parameters
        laser_element_size = mesh_sizes["laser"]
        part_element_size = mesh_sizes["part"]
        baseplate_element_size = mesh_sizes["baseplate"]

        
        baseplate_width, baseplate_length, baseplate_thickness = baseplate_dims
        
        # Use polygon vertices (exclude the last point which is the same as first in Shapely)
        polygon_vertices = []
        coords = list(polygon.exterior.coords)
        for vertex in coords[:-1]:  # Exclude the last point (duplicate of first)
            polygon_vertices.append(vertex)
        
        # Handle interior holes
        hole_vertices_list = []
        if polygon.interiors:
            mpi_print(f"  - Polygon has {len(polygon.interiors)} interior hole(s)")
            for hole in polygon.interiors:
                hole_coords = list(hole.coords)
                hole_vertices = []
                for vertex in hole_coords[:-1]:  # Exclude the last point (duplicate of first)
                    hole_vertices.append(vertex)
                hole_vertices_list.append(hole_vertices)
        
        # Find polygon bounding box to center it
        min_x = min(v[0] for v in polygon_vertices)
        max_x = max(v[0] for v in polygon_vertices)
        min_y = min(v[1] for v in polygon_vertices)
        max_y = max(v[1] for v in polygon_vertices)
        
        polygon_width = max_x - min_x
        polygon_length = max_y - min_y
        
        # Check if polygon fits within baseplate
        if polygon_width > baseplate_width or polygon_length > baseplate_length:
            raise ValueError(f"Polygon size ({polygon_width:.4f} x {polygon_length:.4f}) exceeds baseplate size ({baseplate_width:.4f} x {baseplate_length:.4f})")
        
        
        
        # Validate that centered polygon is within baseplate bounds (centered baseplate)
        baseplate_x_min = -baseplate_width / 2
        baseplate_x_max = baseplate_width / 2
        baseplate_y_min = -baseplate_length / 2
        baseplate_y_max = baseplate_length / 2
        
        # Check exterior vertices
        for vertex in polygon_vertices:
            if vertex[0] < baseplate_x_min or vertex[0] > baseplate_x_max or vertex[1] < baseplate_y_min or vertex[1] > baseplate_y_max:
                raise ValueError(f"Polygon vertex ({vertex[0]:.4f}, {vertex[1]:.4f}) is outside baseplate bounds")
        
        # Check hole vertices
        for hole_vertices in hole_vertices_list:
            for vertex in hole_vertices:
                if vertex[0] < baseplate_x_min or vertex[0] > baseplate_x_max or vertex[1] < baseplate_y_min or vertex[1] > baseplate_y_max:
                    raise ValueError(f"Hole vertex ({vertex[0]:.4f}, {vertex[1]:.4f}) is outside baseplate bounds")
        
        mpi_print(f"Generating mesh:")
        mpi_print(f"  - Baseplate: {baseplate_width:.3f} x {baseplate_length:.3f} x {baseplate_thickness:.3f} m (centered at origin)")
        mpi_print(f"  - Polygon part height: {part_height:.3f} m")
        mpi_print(f"  - Lasered layer depth: {lasered_layer_depth:.4f} m")
        mpi_print(f"  - Element sizes: laser={laser_element_size:.6f}m ({laser_element_size*1000:.3f}mm), part={part_element_size:.6f}m ({part_element_size*1000:.3f}mm), baseplate={baseplate_element_size:.6f}m ({baseplate_element_size*1000:.3f}mm)")
        mpi_print(f"  - Polygon vertices: {len(polygon_vertices)}")
        if hole_vertices_list:
            total_hole_vertices = sum(len(hole) for hole in hole_vertices_list)
            mpi_print(f"  - Hole vertices: {total_hole_vertices} (across {len(hole_vertices_list)} holes)")
        
        # Create baseplate volume - CENTERED AT (0,0)
        baseplate_x_min = -baseplate_width / 2
        baseplate_y_min = -baseplate_length / 2
        baseplate_vol = gmsh.model.occ.addBox(baseplate_x_min, baseplate_y_min, 0, 
                                              baseplate_width, baseplate_length, baseplate_thickness)
        
        # Create polygon points and curves for the exterior boundary
        polygon_points = []
        for i, vertex in enumerate(polygon_vertices):
            point_tag = gmsh.model.occ.addPoint(vertex[0], vertex[1], baseplate_thickness)
            polygon_points.append(point_tag)
        
        # Create exterior polygon edges
        polygon_curves = []
        for i in range(len(polygon_points)):
            next_i = (i + 1) % len(polygon_points)
            curve_tag = gmsh.model.occ.addLine(polygon_points[i], polygon_points[next_i])
            polygon_curves.append(curve_tag)
        
        # Create exterior curve loop
        exterior_loop = gmsh.model.occ.addCurveLoop(polygon_curves)
        
        # Create interior hole loops
        hole_loops = []
        for hole_vertices in hole_vertices_list:
            # Create points for this hole
            hole_points = []
            for vertex in hole_vertices:
                point_tag = gmsh.model.occ.addPoint(vertex[0], vertex[1], baseplate_thickness)
                hole_points.append(point_tag)
            
            # Create curves for this hole
            hole_curves = []
            for i in range(len(hole_points)):
                next_i = (i + 1) % len(hole_points)
                curve_tag = gmsh.model.occ.addLine(hole_points[i], hole_points[next_i])
                hole_curves.append(curve_tag)
            
            # Create curve loop for this hole (note: holes should have opposite orientation)
            hole_loop = gmsh.model.occ.addCurveLoop(hole_curves)
            hole_loops.append(hole_loop)
        
        # Create polygon surface with holes
        all_loops = [exterior_loop] + hole_loops
        polygon_surface = gmsh.model.occ.addPlaneSurface(all_loops)
        
        # Extrude polygon to create part volume (excluding lasered layer)
        part_height_without_lasered = part_height - lasered_layer_depth
        part_extrude = gmsh.model.occ.extrude([(2, polygon_surface)], 0, 0, part_height_without_lasered)
        part_vol = part_extrude[1][1]  # Volume is the second entity in the first result
        
        # Create another polygon surface at the top of the part for lasered layer
        # Exterior boundary points
        lasered_surface_points = []
        lasered_z = baseplate_thickness + part_height_without_lasered
        for vertex in polygon_vertices:
            point_tag = gmsh.model.occ.addPoint(vertex[0], vertex[1], lasered_z)
            lasered_surface_points.append(point_tag)
        
        # Exterior curves
        lasered_curves = []
        for i in range(len(lasered_surface_points)):
            next_i = (i + 1) % len(lasered_surface_points)
            curve_tag = gmsh.model.occ.addLine(lasered_surface_points[i], lasered_surface_points[next_i])
            lasered_curves.append(curve_tag)
        
        # Exterior loop for lasered layer
        lasered_exterior_loop = gmsh.model.occ.addCurveLoop(lasered_curves)
        
        # Create hole loops for lasered layer
        lasered_hole_loops = []
        for hole_vertices in hole_vertices_list:
            # Create points for this hole at lasered layer height
            hole_points = []
            for vertex in hole_vertices:
                point_tag = gmsh.model.occ.addPoint(vertex[0], vertex[1], lasered_z)
                hole_points.append(point_tag)
            
            # Create curves for this hole
            hole_curves = []
            for i in range(len(hole_points)):
                next_i = (i + 1) % len(hole_points)
                curve_tag = gmsh.model.occ.addLine(hole_points[i], hole_points[next_i])
                hole_curves.append(curve_tag)
            
            # Create curve loop for this hole
            hole_loop = gmsh.model.occ.addCurveLoop(hole_curves)
            lasered_hole_loops.append(hole_loop)
        
        # Create lasered surface with holes
        all_lasered_loops = [lasered_exterior_loop] + lasered_hole_loops
        lasered_surface = gmsh.model.occ.addPlaneSurface(all_lasered_loops)
        
        # Extrude lasered surface to create lasered layer volume
        lasered_extrude = gmsh.model.occ.extrude([(2, lasered_surface)], 0, 0, lasered_layer_depth)
        lasered_vol = lasered_extrude[1][1]  # Volume is the second entity in the first result
        
        # Use boolean fragment to ensure proper connectivity
        all_volumes = [(3, baseplate_vol), (3, part_vol), (3, lasered_vol)]
        gmsh.model.occ.fragment(all_volumes, [])
        gmsh.model.occ.synchronize()
        
        # Identify volumes by their center points
        all_vols = gmsh.model.getEntities(3)
        baseplate_vols = []
        part_vols = []
        lasered_vols = []
        
        for vol in all_vols:
            center = gmsh.model.occ.getCenterOfMass(3, vol[1])
            z_center = center[2]
            
            if z_center < baseplate_thickness * 0.9:
                baseplate_vols.append(vol[1])
            elif z_center > baseplate_thickness + part_height_without_lasered + lasered_layer_depth * 0.1:
                lasered_vols.append(vol[1])
            else:
                part_vols.append(vol[1])
        
        # Add physical groups
        if baseplate_vols:
            gmsh.model.addPhysicalGroup(3, baseplate_vols, 1, "Baseplate")
        if part_vols:
            gmsh.model.addPhysicalGroup(3, part_vols, 2, "Part")
        if lasered_vols:
            gmsh.model.addPhysicalGroup(3, lasered_vols, 3, "LaseredLayer")
        
        # Set mesh sizes using volume-based classification instead of point-based
        all_points = gmsh.model.getEntities(0)
        
        # Count points in each region for debugging
        baseplate_points = 0
        part_points = 0
        lasered_points = 0
        unclassified_points = 0
        
        # Create a mapping of points to volumes
        point_to_volume = {}
        
        # For each volume, find all points belonging to it
        for vol_type, vol_list in [("baseplate", baseplate_vols), ("part", part_vols), ("lasered", lasered_vols)]:
            for vol_id in vol_list:
                # Get all boundary surfaces of this volume
                boundary_entities = gmsh.model.getBoundary([(3, vol_id)], combined=False, oriented=False, recursive=True)
                
                # Extract points from boundary entities
                for entity in boundary_entities:
                    if entity[0] == 0:  # Point entity
                        point_id = entity[1]
                        if point_id not in point_to_volume:
                            point_to_volume[point_id] = vol_type
        
        # Assign mesh sizes based on volume classification
        for point in all_points:
            point_id = point[1]
            
            if point_id in point_to_volume:
                vol_type = point_to_volume[point_id]
                if vol_type == "baseplate":
                    gmsh.model.mesh.setSize([point], baseplate_element_size)
                    baseplate_points += 1
                elif vol_type == "part":
                    gmsh.model.mesh.setSize([point], part_element_size)
                    part_points += 1
                elif vol_type == "lasered":
                    gmsh.model.mesh.setSize([point], laser_element_size)
                    lasered_points += 1
            else:
                # Fallback: use coordinate-based classification for unclassified points
                coord = gmsh.model.getValue(0, point_id, [])
                x, y, z = float(coord[0]), float(coord[1]), float(coord[2])
                
                # Use more robust polygon containment with buffer for boundary points
                test_point = Point(x, y)
                # Add small buffer to catch boundary points
                buffered_polygon = polygon.buffer(1e-10)
                point_in_polygon = buffered_polygon.contains(test_point)
                
                if z <= baseplate_thickness + 1e-6:
                    gmsh.model.mesh.setSize([point], baseplate_element_size)
                    baseplate_points += 1
                elif point_in_polygon and z >= lasered_z - 1e-6:
                    gmsh.model.mesh.setSize([point], laser_element_size)
                    lasered_points += 1
                elif point_in_polygon:
                    gmsh.model.mesh.setSize([point], part_element_size)
                    part_points += 1
                else:
                    gmsh.model.mesh.setSize([point], baseplate_element_size)
                    baseplate_points += 1
                unclassified_points += 1
        
        mpi_print(f"Mesh size assignment: {baseplate_points} baseplate points, {part_points} part points, {lasered_points} lasered points")
        if unclassified_points > 0:
            mpi_print(f"  - {unclassified_points} points used fallback coordinate-based classification")
        
        # Also set mesh sizes on curves and surfaces for better control
        # Set mesh sizes on edges belonging to each volume type
        for vol_type, vol_list in [("baseplate", baseplate_vols), ("part", part_vols), ("lasered", lasered_vols)]:
            element_size = {"baseplate": baseplate_element_size, "part": part_element_size, "lasered": laser_element_size}[vol_type]
            
            for vol_id in vol_list:
                # Get boundary surfaces
                surfaces = gmsh.model.getBoundary([(3, vol_id)], oriented=False)
                for surf in surfaces:
                    # Get boundary curves of each surface
                    curves = gmsh.model.getBoundary([surf], oriented=False)
                    for curve in curves:
                        gmsh.model.mesh.setSize(gmsh.model.getBoundary([curve], oriented=False), element_size)
        
        # Find bottom surface for boundary condition
        bottom_surfaces = []
        if baseplate_vols:
            for vol in baseplate_vols:
                surfaces = gmsh.model.getBoundary([(3, vol)], oriented=False)
                for surf in surfaces:
                    center = gmsh.model.occ.getCenterOfMass(2, surf[1])
                    if abs(center[2]) < 1e-6:
                        bottom_surfaces.append(surf[1])
        
        if bottom_surfaces:
            gmsh.model.addPhysicalGroup(2, bottom_surfaces, 4, "BaseplateBottom")
        
        # Generate mesh
        mpi_print("Generating 3D polygon mesh...")
        gmsh.model.mesh.generate(3)
        
        # Write mesh
        gmsh.write(output_file)
        mpi_print(f"Mesh saved to: {output_file}")
        
        # Get statistics
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        num_nodes = len(nodes[0])
        num_elements = sum(len(elem_tags) for elem_tags in elements[1])
        
        mpi_print(f"Mesh statistics: {num_nodes} nodes, {num_elements} elements")
        
        return output_file
        
    except ImportError:
        mpi_print("Gmsh not available. Install with: pip install gmsh")
        return None
    except Exception as e:
        mpi_print(f"Error generating Gmsh polygon mesh: {e}")
        mpi_print(f"Traceback: {traceback.format_exc()}")
        return None
    finally:
        try:
            gmsh.finalize()
        except:
            pass

if __name__ == "__main__":
    mpi_print("=== 3D Thermal Mesh Generator ===")
    try:
        # Generate a shape to test meshing functionality
        from geometry.geometry_samples import generate_shapes
        
        shape = generate_shapes(1, hole=True)[0][0]
        shape = shape.simplify(tolerance=0.00001, preserve_topology=True)
        print(shape)
        part_height = PART_HEIGHT

        mesh_sizes = get_mesh_size_presets('coarse')
        
        polygon_mesh_file = mesh_polygon(
            polygon=shape,
            baseplate_dims=(BASEPLATE_WIDTH, BASEPLATE_LENGTH, BASEPLATE_THICKNESS),
            part_height=part_height,
            lasered_layer_depth=LASER_DIAMETER/2,
            output_file="test_mesh.msh",
            mesh_sizes=mesh_sizes["fine"],
        )
        
        if polygon_mesh_file:
            mpi_print("\nPolygon mesh generation completed successfully!")
            mpi_print("\nLaunching Gmsh GUI for polygon mesh visualization...")
            launch_gmsh_gui(polygon_mesh_file)
        else:
            mpi_print("Polygon mesh generation failed.")
            
    except Exception as e:
        mpi_print(f"Error in polygon mesh test: {e}")
