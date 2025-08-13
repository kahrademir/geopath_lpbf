import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib as mpl
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory to sys.path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)

from shapely.geometry import Polygon, Point
from solver.config import DEFAULT_DT, LASER_DIAMETER, LASER_OVERLAP, SCAN_SPEED

def generate_toolpath_f(
    shape: Polygon,
    speed_func,
    discretization_distance: float,
    time_step: float = DEFAULT_DT,
    pattern: str = "uniraster",
    start_time: float = 0.0,
    raster_angle: float = 0.0,
    laser_diameter: float = LASER_DIAMETER,
    overlap_percentage: float = LASER_OVERLAP,
    add_noise: bool = False,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Generate a laser toolpath for the given polygon, allowing for a variable speed function,
    and using a fixed spatial discretization distance.

    Args:
        shape: The polygon to generate a toolpath for
        speed_func: A function of (time, x, y) or (x, y) or (t) returning the laser speed at that point
        discretization_distance: The fixed spatial step between points (in meters)
        time_step: The fallback time step between points in seconds (used if needed)
        pattern: The pattern to use ("uniraster" only for now)
        start_time: The starting time
        raster_angle: The raster scanning angle in degrees (0° = along x-axis, 90° = along y-axis)
        laser_diameter: The diameter of the laser beam
        overlap_percentage: The desired overlap between laser passes (0.0 = no overlap, 0.5 = 50% overlap)
        add_noise: Whether to add noise to the toolpath points
        noise_std: Standard deviation of the noise to add

    Returns:
        NumPy array of shape (n_points, 3) with [time, x_pos, y_pos] points
    """
    if pattern != "uniraster":
        raise ValueError("generate_toolpath_f currently only supports 'uniraster' pattern.")

    # Get vertices from Shapely polygon exterior coordinates
    vertices = np.array(list(shape.exterior.coords)[:-1])  # Exclude the duplicate last point
    centroid = np.array(shape.centroid.coords[0])

    # Convert angle to radians
    angle_rad = np.radians(raster_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Create rotation matrix for transforming to raster-aligned coordinate system
    rotation_matrix = np.array([[cos_angle, sin_angle],
                                [-sin_angle, cos_angle]])

    # Create inverse rotation matrix for transforming back
    inv_rotation_matrix = np.array([[cos_angle, -sin_angle],
                                    [sin_angle, cos_angle]])

    # Transform vertices and centroid to raster-aligned coordinate system
    transformed_vertices = np.dot(vertices, rotation_matrix.T)
    transformed_shape = Polygon(transformed_vertices)

    min_x, min_y, max_x, max_y = transformed_shape.bounds

    # Compute raster spacing based on laser diameter and overlap
    effective_diameter = laser_diameter * (1.0 - overlap_percentage)
    if effective_diameter <= 0:
        effective_diameter = laser_diameter * 0.1  # fallback to avoid zero division

    # Generate raster lines (horizontal in transformed space)
    y_vals = []
    y = min_y + effective_diameter / 2.0
    while y <= max_y:
        y_vals.append(y)
        y += effective_diameter

    toolpath = []
    current_time = start_time
    direction = 1  # Alternate scan direction for each line

    for i, y in enumerate(y_vals):
        # Find intersections of the horizontal line y = const with the polygon
        line = np.array([[min_x - 1.0, y], [max_x + 1.0, y]])
        # Get all intersections with polygon edges
        intersections = []
        n = len(transformed_vertices)
        for j in range(n):
            p1 = transformed_vertices[j]
            p2 = transformed_vertices[(j + 1) % n]
            # Check if the edge crosses the raster line
            if (p1[1] - y) * (p2[1] - y) <= 0 and p1[1] != p2[1]:
                # Linear interpolation to find intersection x
                x = p1[0] + (y - p1[1]) * (p2[0] - p1[0]) / (p2[1] - p1[1])
                intersections.append(x)
        if len(intersections) < 2:
            continue  # No valid intersection for this line
        intersections = sorted(intersections)
        # Pair up intersections (assume simple polygon)
        for k in range(0, len(intersections) - 1, 2):
            x_start, x_end = intersections[k], intersections[k + 1]
            # Always scan in the same direction (unidirectional: left to right)
            if x_end < x_start:
                x_start, x_end = x_end, x_start
            x_vals = np.arange(x_start, x_end + discretization_distance, discretization_distance)
            if len(x_vals) == 0:
                x_vals = np.array([x_start, x_end])
            for idx, x in enumerate(x_vals):
                # Transform back to original coordinates
                pt = np.dot([x, y], inv_rotation_matrix.T)
                if add_noise:
                    pt = pt + np.random.normal(0, noise_std, size=2)
                # Check if inside original shape
                if shape.contains(Point(pt[0], pt[1])):
                    # Determine speed at this point
                    speed = speed_func(current_time, pt[0], pt[1])
                    # Avoid zero or negative speed
                    if speed <= 0:
                        speed = 1e-6
                    toolpath.append([current_time, pt[0], pt[1]])
                    # Advance time based on distance to next point and local speed
                    if idx < len(x_vals) - 1:
                        # Next point in this segment
                        x_next = x_vals[idx + 1]
                        pt_next = np.dot([x_next, y], inv_rotation_matrix.T)
                        dist = np.linalg.norm(pt_next - pt)
                        current_time += dist / speed
                    else:
                        # End of line, just increment by time_step
                        current_time += time_step
        # No direction flip: always scan in the same direction (unidirectional)

    return np.array(toolpath)





def generate_toolpath(
    shape: Polygon,
    time_step: float = DEFAULT_DT,
    speed: float = SCAN_SPEED,
    pattern: str = "raster",
    start_time: float = 0.0,
    raster_angle: float = 0.0,
    laser_diameter: float = LASER_DIAMETER,
    overlap_percentage: float = LASER_OVERLAP,
    add_noise: bool = False,
    noise_std: float = 0.01
) -> np.ndarray:
    """
    Generate a laser toolpath for the given polygon.
    
    Args:
        shape: The polygon to generate a toolpath for
        time_step: The fixed time step between points in seconds
        speed: The speed of the laser in units/second
        pattern: The pattern to use (only "raster", "uniraster", "spiral_out", "spiral_in" are supported)
        start_time: The starting time
        raster_angle: The raster scanning angle in degrees (0° = along x-axis, 90° = along y-axis)
        laser_diameter: The diameter of the laser beam
        overlap_percentage: The desired overlap between laser passes (0.0 = no overlap, 0.5 = 50% overlap)
        
    Returns:
        NumPy array of shape (n_points, 3) with [time, x_pos, y_pos] points
    """
    # Get vertices from Shapely polygon exterior coordinates
    vertices = np.array(list(shape.exterior.coords)[:-1])  # Exclude the duplicate last point
    centroid = np.array(shape.centroid.coords[0])

    # Convert angle to radians
    angle_rad = np.radians(raster_angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Create rotation matrix for transforming to raster-aligned coordinate system
    rotation_matrix = np.array([[cos_angle, sin_angle],
                               [-sin_angle, cos_angle]])
    
    # Create inverse rotation matrix for transforming back
    inv_rotation_matrix = np.array([[cos_angle, -sin_angle],
                                   [sin_angle, cos_angle]])
    
    # Transform vertices and centroid to raster-aligned coordinate system
    transformed_vertices = np.dot(vertices, rotation_matrix.T)
    transformed_centroid = np.dot(centroid, rotation_matrix.T)
    
    # Get bounding box in transformed coordinate system
    min_x = np.min(transformed_vertices[:, 0])
    min_y = np.min(transformed_vertices[:, 1])
    max_x = np.max(transformed_vertices[:, 0])
    max_y = np.max(transformed_vertices[:, 1])
    
    width = max_x - min_x
    height = max_y - min_y
    
    toolpath = []
    current_time = start_time
    
    # Calculate scan resolution based on laser diameter and desired overlap
    scan_resolution = laser_diameter * (1.0 - overlap_percentage)
    
    if pattern == "raster":
        y = min_y
        direction = 1  # Start moving right
        
        # Calculate step size along each line based on speed and time_step
        step_size = speed * time_step
        
        while y <= max_y:
            row_has_points = False
            
            # Check if this row intersects the shape
            for x_check in np.arange(min_x, max_x + step_size, step_size):
                # Transform check point back to original coordinate system
                transformed_point = np.array([x_check, y])
                original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                
                # Use Shapely's contains method with Point object
                if shape.contains(Point(original_point[0], original_point[1])):
                    row_has_points = True
                    break
                    
            if row_has_points:
                if direction > 0:
                    x_coords = np.arange(min_x, max_x + step_size, step_size)
                else:
                    x_coords = np.arange(max_x, min_x - step_size, -step_size)

                for x in x_coords:
                    # Point in transformed coordinate system
                    transformed_point = np.array([x, y])
                    
                    # Transform back to original coordinate system
                    original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                    original_x, original_y = original_point[0], original_point[1]
                    
                    # Use Shapely's contains method with Point object
                    inside = shape.contains(Point(original_x, original_y))
                    
                    # Only add point and advance time if it's inside the shape
                    if inside:
                        toolpath.append([current_time, original_x, original_y])
                        current_time += time_step
                    # Don't advance time when outside - this eliminates pauses at edges/corners
            
            y += scan_resolution
            direction *= -1
    
    elif pattern == "uniraster":
        y = min_y
        
        # Calculate step size along each line based on speed and time_step
        step_size = speed * time_step
        
        while y <= max_y:
            row_has_points = False
            
            # Check if this row intersects the shape
            for x_check in np.arange(min_x, max_x + step_size, step_size):
                # Transform check point back to original coordinate system
                transformed_point = np.array([x_check, y])
                original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                
                # Use Shapely's contains method with Point object
                if shape.contains(Point(original_point[0], original_point[1])):
                    row_has_points = True
                    break
                    
            if row_has_points:
                # Always scan from left to right (direction = 1)
                x_coords = np.arange(min_x, max_x + step_size, step_size)

                for x in x_coords:
                    # Point in transformed coordinate system
                    transformed_point = np.array([x, y])
                    
                    # Transform back to original coordinate system
                    original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                    original_x, original_y = original_point[0], original_point[1]
                    
                    # Use Shapely's contains method with Point object
                    inside = shape.contains(Point(original_x, original_y))
                    
                    # Only add point and advance time if it's inside the shape
                    if inside:
                        toolpath.append([current_time, original_x, original_y])
                        current_time += time_step
                    # Don't advance time when outside - this eliminates pauses at edges/corners
            
            y += scan_resolution
    elif pattern == "spiral_out":
        # Use the centroid as the center point
        center_x = transformed_centroid[0]
        center_y = transformed_centroid[1]
        
        # Calculate maximum radius from centroid to any corner
        max_radius = max(
            np.sqrt((max_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((max_x - center_x)**2 + (min_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (min_y - center_y)**2)
        )
        
        # Calculate step size along spiral based on speed and time_step
        step_size = speed * time_step
        
        # Spiral parameters
        radius = 0
        angle = 0
        # Increase radius gradually to ensure scan_resolution spacing between turns
        radius_increment = scan_resolution
        
        while radius <= max_radius:
            # Calculate position in spiral
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Check if point is within bounding box
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # Transform point back to original coordinate system
                transformed_point = np.array([x, y])
                original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                original_x, original_y = original_point[0], original_point[1]
                
                # Use Shapely's contains method with Point object
                inside = shape.contains(Point(original_x, original_y))
                
                # Only add point and advance time if it's inside the shape
                if inside:
                    toolpath.append([current_time, original_x, original_y])
                    current_time += time_step
            
            # Calculate next angle increment based on current radius
            # Use step_size to maintain constant linear speed
            if radius > 0:
                angle_increment = step_size / radius
            else:
                angle_increment = 1.0  # Large increment for center point
            
            angle += angle_increment
            
            # Increase radius gradually
            radius += radius_increment * angle_increment / (2 * np.pi)
        
    elif pattern == "spiral_in":
        # Use the centroid as the center point
        center_x = transformed_centroid[0]
        center_y = transformed_centroid[1]
        
        # Calculate maximum radius from centroid to any corner
        max_radius = max(
            np.sqrt((max_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (max_y - center_y)**2),
            np.sqrt((max_x - center_x)**2 + (min_y - center_y)**2),
            np.sqrt((min_x - center_x)**2 + (min_y - center_y)**2)
        )
        
        # Calculate step size along spiral based on speed and time_step
        step_size = speed * time_step
        
        # Spiral parameters - start from outside and spiral in
        radius = max_radius
        angle = 0
        # Decrease radius gradually to ensure scan_resolution spacing between turns
        radius_decrement = scan_resolution
        
        while radius >= 0:
            # Calculate position in spiral
            x = center_x + radius * np.cos(angle)
            y = center_y + radius * np.sin(angle)
            
            # Check if point is within bounding box
            if min_x <= x <= max_x and min_y <= y <= max_y:
                # Transform point back to original coordinate system
                transformed_point = np.array([x, y])
                original_point = np.dot(transformed_point, inv_rotation_matrix.T)
                original_x, original_y = original_point[0], original_point[1]
                
                # Use Shapely's contains method with Point object
                inside = shape.contains(Point(original_x, original_y))
                
                # Only add point and advance time if it's inside the shape
                if inside:
                    toolpath.append([current_time, original_x, original_y])
                    current_time += time_step
            
            # Calculate next angle increment based on current radius
            # Use step_size to maintain constant linear speed
            if radius > 0:
                angle_increment = step_size / radius
            else:
                angle_increment = 1.0  # Large increment for center point
            
            angle += angle_increment
            
            # Decrease radius gradually
            radius -= radius_decrement * angle_increment / (2 * np.pi)
    else:
        raise ValueError(f"Unsupported pattern: {pattern}. Only 'raster' pattern is supported.")
        
    return np.array(toolpath)

def calculate_toolpath_stats(toolpath: np.ndarray) -> dict:
    """
    Calculate statistics for the given toolpath.

    Args:
        toolpath: The toolpath to analyze (np.ndarray of shape (n_points, 3)).

    Returns:
        A dictionary with toolpath statistics.
    """
    if toolpath is None or toolpath.size == 0:
        return {
            "total_time": 0,
            "total_distance": 0,
            "num_points": 0
        }

    total_time = toolpath[-1, 0] - toolpath[0, 0]
    total_distance = 0.0

    # Calculate total_distance
    if toolpath.shape[0] > 1:
        diffs = toolpath[1:, 1:3] - toolpath[:-1, 1:3]
        segment_lengths = np.linalg.norm(diffs, axis=1)
        total_distance = np.sum(segment_lengths)

    return {
        "total_time": total_time,
        "total_distance": total_distance,
        "num_points": toolpath.shape[0]
    }

def plot_toolpath(toolpath: np.ndarray, shape: Polygon):
    
    
    import matplotlib.pyplot as plt

    # Get exterior coordinates
    exterior_coords = np.array(shape.exterior.coords)
    # Get interior (hole) coordinates, if any
    interiors = [np.array(interior.coords) for interior in shape.interiors]

    plt.figure(figsize=(10, 8))
    # Fill the inside of the shape with light gray
    from matplotlib.patches import Polygon as MplPolygon

    # Create a matplotlib Polygon patch for the exterior
    patch = MplPolygon(exterior_coords, closed=True, facecolor='#f5f5f5', edgecolor='none', zorder=0)
    plt.gca().add_patch(patch)

    # If there are holes, fill them with white to "cut out" from the gray
    for interior in interiors:
        hole_patch = MplPolygon(interior, closed=True, facecolor='white', edgecolor='none', zorder=1)
        plt.gca().add_patch(hole_patch)

    # Plot toolpath points
    sc = plt.scatter(toolpath[:, 1], toolpath[:, 2], c=toolpath[:, 0], s=10, cmap='viridis', label="Toolpath Points", zorder=2)
    plt.colorbar(sc, label="Time (s)")

    # Plot exterior boundary
    plt.plot(exterior_coords[:, 0], exterior_coords[:, 1], 'k-', linewidth=2, label="Exterior", zorder=3)

    # Plot interior boundaries (holes), if any, using the same style as exterior
    for interior in interiors:
        plt.plot(interior[:, 0], interior[:, 1], 'k-', linewidth=2, zorder=4)

    plt.title(f"Raster angle: {45}° - {len(toolpath)} points")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis('equal')

    # Set plot limits based on actual shape bounds with some margin
    bounds = shape.bounds
    margin = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
    plt.xlim(bounds[0] - margin, bounds[2] + margin)
    plt.ylim(bounds[1] - margin, bounds[3] + margin)

    # plt.legend()
    plt.show()

if __name__ == "__main__":
    from geometry.geometry_samples import generate_shapes
    import matplotlib.pyplot as plt
    
    shape = generate_shapes(1, min_perimeter_area_ratio=5000, min_area=3e-6)[0]
    
    # Print shape info for debugging
    print(f"Shape bounds: {shape.bounds}")
    print(f"Shape area: {shape.area}")
    
    # Use appropriate parameters for millimeter scale
    # Laser diameter should be much smaller than the shape (e.g., 0.1mm = 0.0001m)
    toolpath = generate_toolpath(
        shape,
        pattern="uniraster",
        raster_angle=45,
        speed=2
    )
    
    if len(toolpath) > 0:
        plot_toolpath(toolpath, shape)
        
        # Print toolpath stats
        stats = calculate_toolpath_stats(toolpath)
        print(f"Toolpath stats: {stats}")
    else:
        print("No toolpath generated")
        print("This might be due to:")
        print("- Laser diameter too large compared to shape")
        print("- Shape too small or invalid")
        print("- Scanning parameters not suitable for shape scale")