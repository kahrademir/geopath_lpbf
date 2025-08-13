import math
import numpy as np
from shapely.geometry import Polygon, Point
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
from solver.config import LASER_DIAMETER, DEFAULT_DT, LASER_OVERLAP
from geometry.__archive.geometry import EPSILON
import numba

DEFAULT_NUM_RAYS = 72
DEFAULT_ALPHA = 0.5

# Common weighting functions for neighbor-based calculations
WEIGHT_FUNCTIONS = {
    'inverse': lambda distances, epsilon=1e-10: 1.0 / (distances + epsilon),
    'inverse_square': lambda distances, epsilon=1e-10: 1.0 / (distances + epsilon) ** 2,
    'gaussian': lambda distances, sigma=0.01: np.exp(-distances ** 2 / (2 * sigma ** 2)),
    'exponential': lambda distances, lam=0.01: np.exp(-distances / lam),

    'cutoff': lambda distances, r=0.01: (distances < r).astype(float),
}

def calculate_gri(point: np.ndarray, shape: Polygon, num_rays: int = DEFAULT_NUM_RAYS, alpha: float = DEFAULT_ALPHA) -> float:
    """
    Calculates the Geometric Resistance Index (GRI) for a given point.
    
    Args:
        point: The [x, y] coordinates to calculate GRI for
        shape: The shape object (polygon)
        num_rays: Number of rays to cast from the point
        alpha: The exponent for the distance calculation
        
    Returns:
        float: The calculated GRI value (0 to 1)
    """
    if not shape.is_inside(point):
        return 0.0  # Zero for points outside the shape (not important)
    
    angle_increment = 2 * np.pi / num_rays
    sum_distances = 0.0
    
    for i in range(num_rays):
        angle = i * angle_increment
        direction = np.array([np.cos(angle), np.sin(angle)])
        distance = shape.intersect_ray(point, direction)
        
        if distance > 0 and distance != float('inf'):
            sum_distances += distance ** alpha  # Simple sum of distances raised to alpha
    
    # Invert and normalize so larger sum (more open space) gives lower GRI
    if sum_distances <= EPSILON:
        return 1.0  # If all rays fail, it's a highly constrained point
    
    # Simple normalization: larger distances mean lower GRI
    # This might need calibration based on shape size
    return 1.0 / (sum_distances ** (1/num_rays)) 

def calculate_gri_for_toolpath(
    toolpath: list[list],
    shape: Polygon,
    num_rays: int = DEFAULT_NUM_RAYS,
    alpha: float = DEFAULT_ALPHA
) -> list[float]:
    """
    Calculates GRI for each point in the laser toolpath.

    Args:
        toolpath: List of [time, x_pos, y_pos] points.
        shape: The polygon object.
        num_rays: Number of rays for GRI calculation.
        alpha: Alpha parameter for GRI calculation.

    Returns:
        List of GRI values corresponding to each toolpath point.
    """
    gri_values = []
    for t, x, y in toolpath:
        point = np.array([x, y])
        gri = calculate_gri(point, shape, num_rays, alpha)
        gri_values.append(gri)
    return gri_values

@numba.njit
def calculate_distances_to_ring(points, ring_coords, epsilon):
    vertices = ring_coords[:-1]  # Already a NumPy array
    n_vertices = len(vertices)
    n_points = len(points)
    ring_distances = np.full(n_points, np.inf)
    for i in range(n_vertices):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n_vertices]
        edge = v2 - v1
        edge_length_sq = np.dot(edge, edge)
        if edge_length_sq < epsilon:
            for j in range(n_points):
                dist_to_v1 = np.linalg.norm(points[j] - v1)
                if dist_to_v1 < ring_distances[j]:
                    ring_distances[j] = dist_to_v1
            continue
        for j in range(n_points):
            point_vector = points[j] - v1
            t = np.dot(point_vector, edge) / edge_length_sq
            t = min(max(t, 0.0), 1.0)
            closest_point = v1 + t * edge
            edge_distance = np.linalg.norm(points[j] - closest_point)
            if edge_distance < ring_distances[j]:
                ring_distances[j] = edge_distance
    return ring_distances

def calculate_signed_distance_field(
    points: np.ndarray,
    shape: Polygon,
    radius: float = 0.0002
) -> np.ndarray:
    """
    Calculates the distance field for an array of points. Points outside the shape will have a value of 0.
    Since we only evaluate this on laser toolpath points which are always inside the geometry,
    we skip the inside/outside check and return only positive distances.
    
    Considers both exterior boundary and interior holes to find the minimum distance to any boundary.
    
    Args:
        points: Array of shape (n_points, 2) containing [x, y] coordinates
        shape: The Shapely Polygon shape to calculate distances relative to
        
    Returns:
        np.ndarray: Array of positive distances to the nearest boundary for each point
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must be a 2D array with shape (n_points, 2)")
    
    # Initialize distances array
    distances = np.full(len(points), np.inf)
    
    # Calculate distances to exterior boundary
    exterior_coords = np.array(shape.exterior.coords)
    exterior_distances = calculate_distances_to_ring(points, exterior_coords, EPSILON)
    distances = np.minimum(distances, exterior_distances)
    
    # Calculate distances to each interior hole
    for interior in shape.interiors:
        interior_coords = np.array(interior.coords)
        interior_distances = calculate_distances_to_ring(points, interior_coords, EPSILON)
        distances = np.minimum(distances, interior_distances)
    
    # Set distances to 0 for points outside the shape
    for i, pt in enumerate(points):
        if not shape.contains(Point(pt[0], pt[1])):
            distances[i] = 0.0
    
    # Calculate the laplacian of the distance field
    # laplacian = calculate_distance_field_laplacian(points, distances, radius=radius)

    return distances

def calculate_distance_field_laplacian(points: np.ndarray, distances: np.ndarray, radius: float = None) -> np.ndarray:
    """
    Calculate the Laplacian of the distance field using local quadratic surface fitting.
    
    Args:
        points: Array of shape (n_points, 2) containing [x, y] coordinates
        distances: Array of distance values at each point
        radius: Radius for finding neighbors. If None, auto-calculated based on point density
        
    Returns:
        np.ndarray: Laplacian values (∇²d) for each point
    """
    n_points = len(points)
    laplacian = np.zeros(n_points)
    
    # Use radius-based nearest neighbors
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(points)
    neighbor_distances, neighbor_indices = nbrs.radius_neighbors(points)
    
    laplacian_calculated = 0
    
    for i in range(n_points):
        neighbor_idx = neighbor_indices[i]
        
        if len(neighbor_idx) < 6:  # Need at least 6 points for quadratic fit
            continue
            
        # Get neighbor coordinates and distance values
        X = points[neighbor_idx]
        D = distances[neighbor_idx]
        
        # Fit quadratic surface: d = a*x² + b*xy + c*y² + d*x + e*y + f
        x_coords = X[:, 0]
        y_coords = X[:, 1]
        A_quad = np.c_[
            x_coords**2,          # coefficient a
            x_coords * y_coords,  # coefficient b  
            y_coords**2,          # coefficient c
            x_coords,             # coefficient d
            y_coords,             # coefficient e
            np.ones(len(X))       # coefficient f
        ]
        
        try:
            quad_coeffs, _, _, _ = np.linalg.lstsq(A_quad, D, rcond=None)
            # Laplacian = ∇²d = ∂²d/∂x² + ∂²d/∂y² = 2a + 2c
            laplacian[i] = 2 * quad_coeffs[0] + 2 * quad_coeffs[2]
            laplacian_calculated += 1
        except np.linalg.LinAlgError:
            # If quadratic fit fails, laplacian remains 0
            pass
    
    print(f"Distance field Laplacian calculation with radius {radius:.4f}:")
    print(f"  Laplacian calculated for: {laplacian_calculated}/{n_points} points")
    print(f"  Success rate: {laplacian_calculated/n_points*100:.1f}%")
    
    return laplacian

def discrete_laplacian(points: np.ndarray, time_values: np.ndarray, r_g=None, radius=None, laser_diameter: float=LASER_DIAMETER, overlap_percentage: float=LASER_OVERLAP, weight_function=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the discrete Laplacian of a time field using weighted graph Laplacian.
    For each point, considers only points with earlier time values within a specified radius.
    Transforms time values to temperature using a hyperbolic tangent model.
    
    Args:
        points: np.ndarray of shape (n_points, 2) - the point coordinates
        time_values: np.ndarray of shape (n_points,) - the time values
        radius: radius within which to find neighbors (in the same units as points)
               If None, automatically calculate based on raster pattern
        laser_diameter: diameter of the laser beam (for radius calculation)
        overlap_percentage: overlap between adjacent passes (for radius calculation)
        weight_function: function that takes distances and returns weights. 
                        If None, uses default inverse square weighting: 1.0 / (distances + epsilon)**2
    
    Returns:
        laplacian: np.ndarray of shape (n_points,) - Laplacian values (∇²T) using weighted graph Laplacian
    """
    n_points = len(points)
    laplacian = np.zeros(n_points)
    laplacian_2 = np.zeros(n_points)
    
    # Auto-calculate radius if not provided
    scan_resolution = laser_diameter * (1.0 - overlap_percentage)
    if radius is None:
        radius = 1.5 * scan_resolution
    if r_g is None:
        r_g = radius

    # Use radius-based nearest neighbors
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(points)
    distances, indices = nbrs.radius_neighbors(points)
    
    points_processed = 0
    points_skipped = 0
    laplacian_calculated = 0
    
    for i in range(n_points):
        neighbor_idx = indices[i]
        current_time = time_values[i]
        
        # Filter neighbors to only include points with time < current_time
        past_neighbors = [idx for idx in neighbor_idx if time_values[idx] < current_time]
        
        if len(past_neighbors) < 3:  # Need at least 3 points to fit a plane
            points_skipped += 1
            continue

        X = points[past_neighbors]
        theta_star = 3200
        theta_amb = 1350
        a = 400
        xi = current_time - time_values[past_neighbors]
        
        T = (theta_star - theta_amb) * np.tanh(a*xi)/a/xi + theta_amb
        # T = np.exp(-xi**2)

        points_processed += 1
        
        # Calculate Laplacian using weighted graph Laplacian if we have enough points

        # Get distances to past neighbors
        all_distances = distances[i]
        all_indices = indices[i]
        
        # Find distances corresponding to past neighbors
        past_neighbor_distances = []
        for past_idx in past_neighbors:
            # Find the position of past_idx in all_indices
            pos = np.where(all_indices == past_idx)[0]
            if len(pos) > 0:
                past_neighbor_distances.append(all_distances[pos[0]])
        
        if len(past_neighbor_distances) > 0:
            past_neighbor_distances = np.array(past_neighbor_distances)
            
            # Calculate spatial weights (inverse distance with small epsilon to avoid division by zero)
            epsilon = 1e-10
            if weight_function is not None:
                weights = weight_function(past_neighbor_distances)
            else:
                weights = 1.0 / (past_neighbor_distances + epsilon)**2
                # weights = np.exp(-past_neighbor_distances ** 2 / (2 * r_g ** 2))
            
            # Calculate weighted graph Laplacian: L[T](i) = Σ w_ij * (T_i - T_j)
            # Note: Using the transformed temperature values T instead of raw time values
            current_value = theta_amb  # T at current time (xi=0, so tanh(0)/0 -> 1, giving theta_star)
            neighbor_values = T  # Already computed transformed temperatures
            
            laplacian[i] = np.sum(weights * (current_value - neighbor_values))
            laplacian_2[i] = np.sum(weights * laplacian[past_neighbors])


            laplacian_calculated += 1
    
    print(f"Laplacian calculation with radius {radius:.4f}:")
    print(f"  Points processed: {points_processed}")
    print(f"  Points skipped (<3 past neighbors): {points_skipped}")
    print(f"  Laplacian calculated: {laplacian_calculated}")
    print(f"  Success rate: {points_processed/(points_processed+points_skipped)*100:.1f}%")
    print(f"  Laplacian success rate: {laplacian_calculated/points_processed*100:.1f}%" if points_processed > 0 else "  Laplacian success rate: 0.0%")
    
    return laplacian, laplacian_2

def discrete_laplacian_v2(points: np.ndarray, 
                          time_values: np.ndarray, 
                          radius: float=None, 
                          laser_diameter: float=LASER_DIAMETER, 
                          overlap_percentage: float=LASER_OVERLAP, 
                          weight_function=None,
                          factor=None) -> np.ndarray:
    """
    Calculate the discrete Laplacian of a time field using weighted graph Laplacian.
    For each point, considers only points with earlier time values within a specified radius.
    Transforms time values to temperature using a hyperbolic tangent model.
    
    Args:
        points: np.ndarray of shape (n_points, 2) - the point coordinates
        time_values: np.ndarray of shape (n_points,) - the time values
        radius: radius within which to find neighbors (in the same units as points)
               If None, automatically calculate based on raster pattern
        laser_diameter: diameter of the laser beam (for radius calculation)
        overlap_percentage: overlap between adjacent passes (for radius calculation)
        weight_function: function that takes distances and returns weights. 
                        If None, uses default inverse square weighting: 1.0 / (distances + epsilon)**2
        factor: factor to multiply the radius by. If None, no factor is applied.
    Returns:
        laplacian: np.ndarray of shape (n_points,) - Laplacian values (∇²T) using weighted graph Laplacian
    """
    n_points = len(points)
    laplacian = np.zeros(n_points)

    # Auto-calculate radius if not provided
    scan_resolution = laser_diameter * (1.0 - overlap_percentage)
    if radius is None:
        radius = 1.5 * scan_resolution

    if factor is not None:
        radius = radius * factor

    # Use radius-based nearest neighbors
    nbrs = NearestNeighbors(radius=radius, algorithm='auto').fit(points)
    distances, indices = nbrs.radius_neighbors(points)
    
    points_processed = 0
    points_skipped = 0
    laplacian_calculated = 0
    
    for i in range(n_points):
        neighbor_idx = indices[i]
        current_time = time_values[i]
        
        # Filter neighbors to only include points with time < current_time
        past_neighbors = [idx for idx in neighbor_idx if time_values[idx] < current_time]
        # past_neighbors = neighbor_idx
   
        T = current_time - time_values[past_neighbors]

        points_processed += 1

        # Get distances to past neighbors
        all_distances = distances[i]
        all_indices = indices[i]
        
        # Find distances corresponding to past neighbors
        past_neighbor_distances = []
        for past_idx in past_neighbors:
            # Find the position of past_idx in all_indices
            pos = np.where(all_indices == past_idx)[0]
            if len(pos) > 0:
                past_neighbor_distances.append(all_distances[pos[0]])
        
        if len(past_neighbor_distances) > 0:
            past_neighbor_distances = np.array(past_neighbor_distances)
            
            # Calculate spatial weights (inverse distance with small epsilon to avoid division by zero)
            epsilon = 1e-10
            if weight_function is not None:
                weights = weight_function(past_neighbor_distances)
            else:
                weights = 1.0 / (past_neighbor_distances + epsilon)**2
                # weights = np.exp(-past_neighbor_distances ** 2 / (2 * r_g ** 2))
            
            # Calculate weighted graph Laplacian: L[T](i) = Σ w_ij * (T_i - T_j)
            # if i > 0:
            #     laplacian[i] = np.sum(weights * T * np.abs(laplacian[i-1]))
            # else:
            laplacian[i] = np.sum(weights * T)

            laplacian_calculated += 1
    
    print(f"Laplacian calculation with radius {radius:.4f}:")
    print(f"  Points processed: {points_processed}")
    print(f"  Points skipped (<3 past neighbors): {points_skipped}")
    print(f"  Laplacian calculated: {laplacian_calculated}")
    print(f"  Success rate: {points_processed/(points_processed+points_skipped)*100:.1f}%")
    print(f"  Laplacian success rate: {laplacian_calculated/points_processed*100:.1f}%" if points_processed > 0 else "  Laplacian success rate: 0.0%")
    
    return laplacian

def calculate_optimal_radius(points: np.ndarray, target_neighbors: int, laser_diameter: float, overlap_percentage: float) -> float:

    """
    Calculate an optimal radius based on the raster pattern spacing to ensure
    points from adjacent paths are included in the neighborhood.
    
    Args:
        points: np.ndarray of shape (n_points, 2) - the point coordinates
        target_neighbors: target number of neighbors (including self)
        laser_diameter: diameter of the laser beam
        overlap_percentage: overlap between adjacent passes (0.0 to 1.0)
    
    Returns:
        float: optimal radius that captures adjacent raster paths
    """
    # Calculate average spacing between consecutive points along the same path
    if len(points) > 1:
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        avg_spacing = np.mean(distances)
    else:
        avg_spacing = 0.1  # Default fallback
    
    # Calculate effective raster spacing (distance between adjacent paths)
    # This accounts for laser diameter and overlap
    effective_raster_spacing = laser_diameter * (1 - overlap_percentage)
    
    # The radius should be at least 1.5 times the raster spacing to capture
    # points from adjacent paths, plus some buffer for points along the same path
    min_radius_for_adjacent_paths = effective_raster_spacing * 1.8
    
    # Also consider the target number of neighbors for statistical robustness
    radius_for_target_neighbors = avg_spacing * np.sqrt(target_neighbors / np.pi)
    
    # Use the larger of the two to ensure both criteria are met
    optimal_radius = max(min_radius_for_adjacent_paths, radius_for_target_neighbors)
    
    print(f"Average point spacing along path: {avg_spacing:.4f}")
    print(f"Effective raster spacing: {effective_raster_spacing:.4f}")
    print(f"Minimum radius for adjacent paths: {min_radius_for_adjacent_paths:.4f}")
    print(f"Radius for target neighbors: {radius_for_target_neighbors:.4f}")
    print(f"Selected optimal radius: {optimal_radius:.4f}")
    
    return optimal_radius