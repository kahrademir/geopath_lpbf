#!/usr/bin/env python3
"""
Configuration file for 3D thermal analysis simulation
Contains all simulation parameters, material properties, and mesh settings
"""

# Part geometry (m)
# PART_WIDTH and PART_LENGTH removed - now defined by Polygon shape
PART_HEIGHT = 0.001     # Height (z-dimension) of the lasered layer

# Baseplate dimensions (m) - fixed, generous size to accommodate various polygon shapes
BASEPLATE_WIDTH = 0.025    # 25mm baseplate width
BASEPLATE_LENGTH = 0.025   # 25mm baseplate length
BASEPLATE_THICKNESS = 0.003             

# Laser parameters
LASER_POWER = 10                # (W) default is 10 W
LASER_DIAMETER = 0.0001         # m
LASER_RADIUS = LASER_DIAMETER / 2
LASER_OVERLAP = 0.25
SCAN_SPEED = 1                # m/s
ABSORPTION_COEFF = 1            # absorption coefficient

# Lasered layer depth
LASERED_LAYER_DEPTH = LASER_RADIUS

# Material properties (stainless steel 316L)
DENSITY = 7900.0                # kg/m³
SPECIFIC_HEAT = 500.0           # J/kg·K
THERMAL_CONDUCTIVITY = 20.0     # W/m·K

# Latent heat of fusion parameters
MELTING_TEMP = 1673.15          # K
LATENT_HEAT = 270000.0          # J/kg
MELTING_RANGE = 100.0           # K

# Temperature boundary conditions
AMBIENT_TEMP = 873.15           # K
BASEPLATE_TEMP = AMBIENT_TEMP   # K

# Time stepping
DEFAULT_DT = LASER_DIAMETER * 0.25 / SCAN_SPEED              # s - set for ~25% laser overlap
COOLING_DT_MULTIPLIER = 100     # cooling timestep = dt * this multiplier
COOLING_TIME = 0.01              # s

# PETSc solver options
PETSC_OPTIONS = {
    "ksp_type": "cg",
    "pc_type": "hypre", 
    "pc_hypre_type": "boomeramg",
    "ksp_rtol": 1e-8,
    "ksp_atol": 1e-12,
    "ksp_max_it": 1000
} 

PRESET_MESH_SIZES = {
        'ultra_fine': {
            'laser': LASER_DIAMETER / 8,      # Very fine mesh in laser region
            'part': LASER_DIAMETER * 5,       # Fine mesh in part
            'baseplate': LASER_DIAMETER * 20   # Medium mesh in baseplate
        },
        'fine': {
            'laser': LASER_DIAMETER / 4,      # Fine mesh in laser region (default)
            'part': LASER_DIAMETER * 10,       # Medium mesh in part (default)
            'baseplate': LASER_DIAMETER * 50   # Coarse mesh in baseplate (default)
        },
        'medium': {
            'laser': LASER_DIAMETER / 2,      # Medium-fine mesh in laser region
            'part': LASER_DIAMETER * 10,       # Medium mesh in part
            'baseplate': LASER_DIAMETER * 50   # Coarser mesh in baseplate
        },
        'coarse': {
            'laser': LASER_DIAMETER,          # Coarse mesh in laser region
            'part': LASER_DIAMETER * 10,       # Coarse mesh in part
            'baseplate': LASER_DIAMETER * 50   # Very coarse mesh in baseplate
        },
        'ultra_coarse': {
            'laser': LASER_DIAMETER * 4,      # Very coarse mesh in laser region
            'part': LASER_DIAMETER * 10,       # Very coarse mesh in part
            'baseplate': LASER_DIAMETER * 50   # Very coarse mesh in baseplate
        }
    }

def get_mesh_size_presets(preset: str = 'coarse'):
    """
    Get predefined mesh size configurations for different simulation needs
    
    Returns:
        dict: Dictionary of preset configurations
    """
    if preset not in PRESET_MESH_SIZES:
        raise ValueError(f"Invalid preset: {preset}. Valid presets are: ultra_fine, fine, medium, coarse, ultra_coarse")
    
    return PRESET_MESH_SIZES[preset]

def validate_polygon_geometry(polygon):
    """
    Validate that a polygon fits within the baseplate bounds with appropriate margins
    
    Args:
        polygon: Polygon object defining the part shape
        
    Returns:
        tuple: (is_valid, error_message)
    """
    import numpy as np
    
    # Get polygon bounding box
    vertices = np.array([[v[0], v[1]] for v in polygon.exterior.coords])
    min_x, min_y = vertices.min(axis=0) 
    max_x, max_y = vertices.max(axis=0)
    
    # Calculate polygon dimensions
    polygon_width = max_x - min_x
    polygon_length = max_y - min_y
    
    # Add safety margins (1mm on each side)
    margin = 0.001
    required_baseplate_width = polygon_width + 2 * margin
    required_baseplate_length = polygon_length + 2 * margin
    
    # Check if polygon fits within baseplate (with centered coordinate system)
    if required_baseplate_width > BASEPLATE_WIDTH:
        return False, f"Polygon width ({polygon_width*1000:.1f}mm) + margins exceeds baseplate width ({BASEPLATE_WIDTH*1000:.1f}mm)"
    
    if required_baseplate_length > BASEPLATE_LENGTH:
        return False, f"Polygon length ({polygon_length*1000:.1f}mm) + margins exceeds baseplate length ({BASEPLATE_LENGTH*1000:.1f}mm)"
    
    # Check if polygon is positioned to fit within centered baseplate bounds
    baseplate_half_width = BASEPLATE_WIDTH / 2
    baseplate_half_length = BASEPLATE_LENGTH / 2
    
    if min_x < -baseplate_half_width or max_x > baseplate_half_width:
        return False, f"Polygon X coordinates ({min_x*1000:.1f}mm to {max_x*1000:.1f}mm) exceed baseplate bounds (±{baseplate_half_width*1000:.1f}mm)"
    
    if min_y < -baseplate_half_length or max_y > baseplate_half_length:
        return False, f"Polygon Y coordinates ({min_y*1000:.1f}mm to {max_y*1000:.1f}mm) exceed baseplate bounds (±{baseplate_half_length*1000:.1f}mm)"
    
    return True, f"Polygon fits within centered baseplate. Dimensions: {polygon_width*1000:.1f}mm x {polygon_length*1000:.1f}mm. Bounds: X=[{min_x*1000:.1f}, {max_x*1000:.1f}]mm, Y=[{min_y*1000:.1f}, {max_y*1000:.1f}]mm"

