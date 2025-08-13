#!/usr/bin/env python3
"""
Heat source and laser toolpath functions for 3D thermal analysis
"""

import numpy as np
from solver.config import *

def laser_heat_source_3d_toolpath(x, t, toolpath, power=LASER_POWER, radius=LASER_RADIUS, absorption=ABSORPTION_COEFF):
    """
    Time-dependent 3D laser heat source following a toolpath
    
    Args:
        x: Spatial coordinates (3D array)
        t: Current time  
        toolpath: List of [time, x_pos, y_pos] points
        power: Laser power (W)
        radius: Laser radius (m)
        absorption: Absorption coefficient
    
    Returns:
        heat_source: Heat source intensity (W/m³)
    """
    # Initialize output array with zeros
    heat_source = np.zeros(x.shape[1])
    
    if len(toolpath) == 0:
        return heat_source
    
    # Check if current time is within toolpath duration
    toolpath_start_time = toolpath[0][0]
    toolpath_end_time = toolpath[-1][0]
    
    if t < toolpath_start_time or t > toolpath_end_time:
        # Laser is off before toolpath starts or after it ends
        return heat_source
    
    # Find bracketing toolpath points for interpolation
    for i in range(len(toolpath) - 1):
        t0, x0, y0 = toolpath[i]
        t1, x1, y1 = toolpath[i+1]
        if t0 <= t <= t1:
            # Linear interpolation
            if t1 == t0:
                alpha = 0.0
            else:
                alpha = (t - t0) / (t1 - t0)
            laser_x = x0 + alpha * (x1 - x0)
            laser_y = y0 + alpha * (y1 - y0)
            break
    else:
        # Use last point if time is at the end of toolpath
        laser_x, laser_y = toolpath[-1][1], toolpath[-1][2]

    # Laser is always on during toolpath duration
    # Distance from laser center
    # x has shape (3, N) where x[0] is all x-coords, x[1] is all y-coords, x[2] is all z-coords
    distance_sq = (x[0] - laser_x)**2 + (x[1] - laser_y)**2
    
    # Calculate the target depth
    target_depth = BASEPLATE_THICKNESS + PART_HEIGHT
    
    # Gaussian distribution with exponential decay in depth
    lateral_intensity = np.exp(-distance_sq / (2 * radius**2))
    depth_attenuation = np.exp(-np.abs(x[2] - target_depth) / (2 * radius))
    
    # Heat source intensity (W/m³)
    intensity = (absorption * power * lateral_intensity * depth_attenuation) / (np.pi * radius**2 * radius)
    
    heat_source = intensity
    
    return heat_source 