#!/usr/bin/env python3
"""
Geometry Sample Generator

This module contains functions for generating various geometric shapes using the exact methods
and shapes from shape_randomization_exploration.py for thermal simulation studies.
"""
# %%
import numpy as np
import random
import sys
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
from shapely.prepared import prep

# Add path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Imports
from shapely.geometry import Polygon, Point, LinearRing
# Use shapely.vectorized if available for fast point-in-polygon
try:
    from shapely import vectorized as _shp_vectorized
except Exception:  # pragma: no cover
    _shp_vectorized = None
from shapely.affinity import rotate, translate

# Configuration constants
MAX_RADIUS = 2.5e-3  # 2.5mm maximum radius constraint
TOLERANCE = MAX_RADIUS * 0.005
SPACE_SIZE = MAX_RADIUS * 2  # Define the space size for random positioning
MIN_WALL_THICKNESS = MAX_RADIUS * 0.075  # Minimum wall thickness (2% of max radius)

def create_square(size: float, hollowness: float = 0.0) -> Polygon:
    """
    Create a square with the given size and optional hollowness.
    
    Args:
        size: Length of the square's side
        hollowness: Ratio from 0.0 (solid) to 1.0 (maximum hollow)
        
    Returns:
        Shapely Polygon representing the square (hollow if hollowness > 0)
    """
    half_size = size / 2
    # Create base square
    base_square = Polygon([
        (-half_size, -half_size),
        (half_size, -half_size),
        (half_size, half_size),
        (-half_size, half_size)
    ])
    
    # If hollowness is 0, return solid square
    if hollowness <= 0.0:
        return base_square
    
    # Calculate inner square size based on hollowness and minimum wall thickness
    max_inner_size = size - 2 * MIN_WALL_THICKNESS
    if max_inner_size <= 0:
        return base_square  # Too small to hollow out
    
    inner_size = max_inner_size * hollowness
    inner_half_size = inner_size / 2
    
    # Create inner square (hole)
    inner_square = Polygon([
        (-inner_half_size, -inner_half_size),
        (inner_half_size, -inner_half_size),
        (inner_half_size, inner_half_size),
        (-inner_half_size, inner_half_size)
    ])
    
    # Create hollow square by subtracting inner from outer
    hollow_square = base_square.difference(inner_square)
    
    return hollow_square

def create_ellipse(width: float, height: float, hollowness: float = 0.0, num_points: int = 32) -> Polygon:
    """
    Create an ellipse approximated as a polygon with the given width and height and optional hollowness.
    
    Args:
        width: Width (horizontal diameter) of the ellipse
        height: Height (vertical diameter) of the ellipse
        hollowness: Ratio from 0.0 (solid) to 1.0 (maximum hollow)
        num_points: Number of points to approximate the ellipse
        
    Returns:
        Shapely Polygon representing the ellips
    """
    # Create ellipse using parametric equations
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    a = width / 2  # semi-major axis
    b = height / 2  # semi-minor axis
    
    x = a * np.cos(angles)
    y = b * np.sin(angles)
    
    coords = list(zip(x, y))
    base_ellipse = Polygon(coords)
    
    # If hollowness is 0, return solid ellipse
    if hollowness <= 0.0:
        return base_ellipse
    
    # Calculate inner ellipse dimensions based on hollowness and minimum wall thickness
    min_dim = min(width, height)
    max_inner_dim = min_dim - 2 * MIN_WALL_THICKNESS
    if max_inner_dim <= 0:
        return base_ellipse  # Too small to hollow out
    
    # Scale both dimensions proportionally to maintain ellipse shape
    scale_factor = (max_inner_dim / min_dim) * hollowness
    inner_width = width * scale_factor
    inner_height = height * scale_factor
    
    # Create inner ellipse (hole)
    inner_a = inner_width / 2
    inner_b = inner_height / 2
    inner_x = inner_a * np.cos(angles)
    inner_y = inner_b * np.sin(angles)
    inner_coords = list(zip(inner_x, inner_y))
    inner_ellipse = Polygon(inner_coords)
    
    # Create hollow ellipse by subtracting inner from outer
    hollow_ellipse = base_ellipse.difference(inner_ellipse)
    
    return hollow_ellipse

def create_triangle(base: float, height: float, hollowness: float = 0.0) -> Polygon:
    """
    Create a triangle with the given base and height and optional hollowness with uniform wall thickness.
    
    Args:
        base: Base length of the triangle
        height: Height of the triangle
        hollowness: Ratio from 0.0 (solid) to 1.0 (maximum hollow)
        
    Returns:
        Shapely Polygon representing the triangle (hollow if hollowness > 0)
        Note: Uses negative buffering for uniform wall thickness around perimeter
    """
    half_base = base / 2
    # Create base triangle
    base_triangle = Polygon([
        (-half_base, -height/2),
        (half_base, -height/2),
        (0, height/2)
    ])
    
    # If hollowness is 0, return solid triangle
    if hollowness <= 0.0:
        return base_triangle
    
    # Calculate wall thickness for uniform inward offset
    # Use the maximum possible inward buffer distance
    min_dim = min(base, height)
    max_inward_distance = (min_dim / 4)  # Conservative max to avoid degeneracy
    
    # Calculate actual wall thickness based on hollowness
    wall_thickness = max_inward_distance * (1.0 - hollowness)
    wall_thickness = max(wall_thickness, MIN_WALL_THICKNESS)  # Enforce minimum
    
    # Use negative buffer to create uniform inward offset
    try:
        inner_triangle = base_triangle.buffer(-wall_thickness)
        
        # Check if buffering resulted in empty geometry (triangle too small)
        if inner_triangle.is_empty or inner_triangle.area <= 0:
            return base_triangle  # Too small to hollow out uniformly
        
        # Create hollow triangle by subtracting buffered inner from outer
        hollow_triangle = base_triangle.difference(inner_triangle)
        
    except Exception as e:
        # If buffering fails, fall back to solid triangle
        return base_triangle
    
    return hollow_triangle

def weighted_random_hollowness(hollowness_bias: float = 0.5) -> float:
    """
    Generate a weighted random hollowness value.
    
    Args:
        hollowness_bias: Bias parameter from 0.0 (prefer solid) to 1.0 (prefer hollow)
        
    Returns:
        Random hollowness value from 0.0 to 1.0, weighted by bias
    """
    # Use a power function to bias the random distribution
    # bias < 0.5 favors solid shapes, bias > 0.5 favors hollow shapes
    base_random = random.random()
    
    if hollowness_bias < 0.5:
        # Bias toward solid (lower values)
        power = 1.0 / (2 * hollowness_bias + 0.1)  # Higher power = more solid shapes
        return base_random ** power
    else:
        # Bias toward hollow (higher values)
        power = 2 * (hollowness_bias - 0.5) + 1  # Higher power = more hollow shapes
        return 1.0 - ((1.0 - base_random) ** power)

def random_transform(shape: Polygon, space_size: float) -> Polygon:
    """
    Apply random rotation and translation to a shape within the defined space.
    
    Args:
        shape: Shapely Polygon to transform
        space_size: Size of the space for random positioning, constrained by MAX_RADIUS
        
    Returns:
        Transformed Shapely Polygon
    """
    # Random rotation between 0 and 360 degrees
    rotation_angle = random.uniform(0, 360)
    shape_rotated = rotate(shape, rotation_angle, origin='centroid')
    
    # Random translation within the space, constrained to keep shapes reasonably close
    x_offset = random.uniform(-space_size/4, space_size/4)
    y_offset = random.uniform(-space_size/4, space_size/4)
    shape_transformed = translate(shape_rotated, xoff=x_offset, yoff=y_offset)
    
    return shape_transformed

def generate_composite_shape(hollowness_bias: float = 0.5, num_union_shapes: int = None, num_subtract_shapes: int = None) -> Polygon:
    """
    Generate a composite shape by combining primitive shapes with random transformations,
    then performing union and subtraction operations.
    
    Args:
        hollowness_bias: Bias parameter from 0.0 (prefer solid) to 1.0 (prefer hollow)
        num_union_shapes: Number of shapes to union together (None for random 2-4)
        num_subtract_shapes: Number of shapes to subtract (None for random 1-2)
    
    Returns:
        Final Shapely Polygon
    """
    # Define size ranges relative to MAX_RADIUS for continuous random sizing
    fillet_percentage = random.uniform(0.025, 0.125)
    
    # Size ranges for each shape type
    min_size_factor = 0.3  # Minimum size as percentage of MAX_RADIUS
    max_size_factor = 0.8  # Maximum size as percentage of MAX_RADIUS
    
    # Determine number of union shapes
    if num_union_shapes is None:
        num_union_shapes = random.randint(2, 4)  # Default: 2-4 random union shapes
    else:
        num_union_shapes = max(1, num_union_shapes)  # Ensure at least 1 union shape
    
    union_shapes = []
    
    for _ in range(num_union_shapes):
        # Randomly select shape type
        shape_type = random.choice(['square', 'ellipse', 'triangle'])
        
        # Generate random hollowness for this shape
        hollowness = weighted_random_hollowness(hollowness_bias)
        
        if shape_type == 'square':
            size = random.uniform(min_size_factor, max_size_factor) * MAX_RADIUS
            shape = create_square(size, hollowness)
        elif shape_type == 'ellipse':
            width = random.uniform(min_size_factor, max_size_factor) * MAX_RADIUS
            height = random.uniform(min_size_factor, max_size_factor) * MAX_RADIUS
            shape = create_ellipse(width, height, hollowness)
        else:  # triangle
            base = random.uniform(min_size_factor, max_size_factor) * MAX_RADIUS
            height = random.uniform(min_size_factor, max_size_factor) * MAX_RADIUS
            shape = create_triangle(base, height, hollowness)
        
        # Apply random transformation
        transformed_shape = random_transform(shape, SPACE_SIZE)
        union_shapes.append(transformed_shape)
    
    # Determine number of subtraction shapes
    if num_subtract_shapes is None:
        num_subtract_shapes = random.randint(1, 2)  # Default: 1-2 random subtraction shapes
    else:
        num_subtract_shapes = max(0, num_subtract_shapes)  # Allow 0 subtraction shapes
    
    subtract_shapes = []
    
    for _ in range(num_subtract_shapes):
        # Randomly select shape type
        shape_type = random.choice(['square', 'ellipse', 'triangle'])
        
        # Use smaller size range for subtraction shapes
        min_subtract_factor = 0.15
        max_subtract_factor = 0.4
        
        # Generate random hollowness for this shape
        hollowness = weighted_random_hollowness(hollowness_bias)
        
        if shape_type == 'square':
            size = random.uniform(min_subtract_factor, max_subtract_factor) * MAX_RADIUS
            shape = create_square(size, hollowness)
        elif shape_type == 'ellipse':
            width = random.uniform(min_subtract_factor, max_subtract_factor) * MAX_RADIUS
            height = random.uniform(min_subtract_factor, max_subtract_factor) * MAX_RADIUS
            shape = create_ellipse(width, height, hollowness)
        else:  # triangle
            base = random.uniform(min_subtract_factor, max_subtract_factor) * MAX_RADIUS
            height = random.uniform(min_subtract_factor, max_subtract_factor) * MAX_RADIUS
            shape = create_triangle(base, height, hollowness)
        
        # Apply random transformation
        transformed_shape = random_transform(shape, SPACE_SIZE)
        subtract_shapes.append(transformed_shape)
    
    # Union all the union shapes with error handling for topology issues
    union_shape = union_shapes[0]
    for shape in union_shapes[1:]:
        try:
            union_shape = union_shape.union(shape)
        except Exception as e:
            print(f"Warning: Union operation failed with error {e}, skipping shape")
            continue
    
    # Subtract all subtraction shapes from the union
    final_shape = union_shape
    for shape in subtract_shapes:
        try:
            final_shape = final_shape.difference(shape)
        except Exception as e:
            print(f"Warning: Difference operation failed with error {e}, skipping shape")
            continue
    
    # Apply buffering to the final composite shape
    fillet_radius = MAX_RADIUS * fillet_percentage
    final_shape = final_shape.buffer(0.5 * fillet_radius).buffer(-1 * fillet_radius).buffer(0.5 * fillet_radius)
    
    # Ensure the result is a single Polygon (handle MultiPolygon if necessary)
    if final_shape.geom_type == 'MultiPolygon':
        # Take the largest polygon by area
        final_shape = max(final_shape.geoms, key=lambda p: p.area)
    elif final_shape.geom_type != 'Polygon':
        # If not a Polygon or MultiPolygon, return an empty polygon
        final_shape = Polygon()
    
    final_shape = final_shape.simplify(tolerance=TOLERANCE, preserve_topology=True)

    return final_shape

def center_polygon_at_origin(polygon: Polygon) -> Polygon:
    """
    Center a polygon at the origin.
    
    Args:
        polygon: Shapely Polygon to center
    """
    # Calculate centroid of the polygon
    centroid = polygon.centroid
    
    # Create translation vector to move centroid to origin
    translation_vector = Point(0, 0) - centroid
    
    # Apply translation to move polygon to origin
    centered_polygon = translate(polygon, xoff=translation_vector.x, yoff=translation_vector.y)
    
    return centered_polygon

def plot_shape(shape: Polygon):
    """
    Plot a single composite shape and display it.
    
    Args:
        shape: Shapely Polygon to plot
    """
    # Create single plot
    fig, ax = plt.subplots(figsize=(6, 6))
    
    if shape.is_empty:
        ax.text(0.5, 0.5, 'Empty', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
    else:
        if shape.geom_type == 'Polygon':
            x, y = shape.exterior.xy
            ax.fill(x, y, 'blue', alpha=0.7)
            ax.plot(x, y, 'k-', linewidth=1.5)
            # Plot holes if any
            for interior in shape.interiors:
                x, y = interior.xy
                ax.fill(x, y, 'white', alpha=1.0)
                ax.plot(x, y, 'k-', linewidth=1.5)
        elif shape.geom_type == 'MultiPolygon':
            for geom in shape.geoms:
                x, y = geom.exterior.xy
                ax.fill(x, y, 'blue', alpha=0.7)
                ax.plot(x, y, 'k-', linewidth=1.5)
    
    ax.set_xlim(-SPACE_SIZE/2, SPACE_SIZE/2)
    ax.set_ylim(-SPACE_SIZE/2, SPACE_SIZE/2)
    ax.set_aspect('equal')
    ax.axis('off')  # Remove axes
    
    plt.show()

def plot_all_shapes(shapes: List[Polygon], save_file: str = None):
    """
    Plot all generated shapes in a grid layout.
    
    Args:
        shapes: List of Shapely Polygons to plot
        save_file: Optional file path to save the plot
    """
    if not shapes:
        print("No shapes to plot")
        return
    
    num_shapes = len(shapes)
    
    # Determine grid layout (try to make it roughly square)
    cols = int(np.ceil(np.sqrt(num_shapes)))
    rows = int(np.ceil(num_shapes / cols))
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single shape case
    if num_shapes == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Plot each shape
    for i, shape in enumerate(shapes):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        if shape.is_empty:
            ax.text(0.5, 0.5, 'Empty', horizontalalignment='center', 
                   verticalalignment='center', transform=ax.transAxes)
        else:
            if shape.geom_type == 'Polygon':
                x, y = shape.exterior.xy
                ax.fill(x, y, alpha=0.7, color='blue')
                ax.plot(x, y, 'k-', linewidth=1.5)
                # Plot holes if any
                for interior in shape.interiors:
                    x, y = interior.xy
                    ax.fill(x, y, alpha=1.0, color='white')
                    ax.plot(x, y, 'k-', linewidth=1.5)
            elif shape.geom_type == 'MultiPolygon':
                for geom in shape.geoms:
                    x, y = geom.exterior.xy
                    ax.fill(x, y, alpha=0.7, color='blue')
                    ax.plot(x, y, 'k-', linewidth=1.5)
        
        # Calculate metrics for title
        perimeter_area_ratio = shape.length / shape.area if shape.area > 0 else float('inf')
        # ax.set_title(f'Shape {i+1}\nArea: {shape.area:.2e}\nP/A: {perimeter_area_ratio:.0f}', 
        #             fontsize=10)
        ax.set_xlim(-SPACE_SIZE/2, SPACE_SIZE/2)
        ax.set_ylim(-SPACE_SIZE/2, SPACE_SIZE/2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    # Hide empty subplots
    for i in range(num_shapes, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_file:
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        print(f"Shapes saved to {save_file}")
    
    plt.show() 

def generate_shapes(num_shapes: int, hole: bool = None, min_area: float = None, max_area: float = None, min_perimeter: float = None, max_perimeter: float = None, max_diameter: float = None, min_perimeter_area_ratio: float = None, max_perimeter_area_ratio: float = None, hollowness_bias: float = 0.5, num_union_shapes: int = None, num_subtract_shapes: int = None, max_attempts: int = 100) -> List[Polygon]:
    """
    Generate multiple composite shapes for visualization with optional constraints.
    
    Args:
        num_shapes: Number of shapes to generate
        hole: If True, only return shapes with interior holes
        min_area: Minimum required area for generated shapes (None for no constraint)
        max_area: Maximum allowed area for generated shapes (None for no constraint)
        min_perimeter: Minimum required perimeter for generated shapes (None for no constraint)
        max_perimeter: Maximum allowed perimeter for generated shapes (None for no constraint)
        max_diameter: Maximum allowed circumscribed diameter for generated shapes (None for no constraint)
        min_perimeter_area_ratio: Minimum perimeter/area ratio to select complex shapes (None for no constraint)
        max_perimeter_area_ratio: Maximum perimeter/area ratio to select complex shapes (None for no constraint)
        hollowness_bias: Bias parameter from 0.0 (prefer solid) to 1.0 (prefer hollow) shapes
        num_union_shapes: Number of shapes to union together (None for random 2-4)
        num_subtract_shapes: Number of shapes to subtract (None for random 1-2, 0 for no subtraction)
        max_attempts: Maximum attempts per shape before giving up on constraints
        
    Returns:
        List of Shapely Polygons that meet the specified constraints
    """
    shapes = []
    for i in range(num_shapes):
        attempts = 0
        while attempts < max_attempts:
            shape = generate_composite_shape(hollowness_bias, num_union_shapes, num_subtract_shapes)
            
            # Check if shape is empty
            if shape.is_empty:
                attempts += 1
                continue
            
            # Check hole constraint
            if hole is not None:
                if hole and not shape.interiors:
                    attempts += 1
                    continue
                elif not hole and shape.interiors:
                    attempts += 1
                    continue
            
            # Check area constraints
            if min_area is not None and shape.area < min_area:
                attempts += 1
                continue
            
            if max_area is not None and shape.area > max_area:
                attempts += 1
                continue
            
            # Check perimeter constraint
            if min_perimeter is not None and shape.length < min_perimeter:
                attempts += 1
                continue
            
            if max_perimeter is not None and shape.length > max_perimeter:
                attempts += 1
                continue
            
            # Check perimeter-to-area ratio for complex shapes
            if min_perimeter_area_ratio is not None or max_perimeter_area_ratio is not None:
                perimeter_area_ratio = shape.length / shape.area if shape.area > 0 else float('inf')
                if min_perimeter_area_ratio is not None and perimeter_area_ratio < min_perimeter_area_ratio:
                    attempts += 1
                    continue
                if max_perimeter_area_ratio is not None and perimeter_area_ratio > max_perimeter_area_ratio:
                    attempts += 1
                    continue
            
            # Check diameter constraint (circumscribed circle diameter)
            if max_diameter is not None:
                # Get bounding box to approximate circumscribed diameter
                minx, miny, maxx, maxy = shape.bounds
                bbox_width = maxx - minx
                bbox_height = maxy - miny
                # Use diagonal of bounding box as approximation of circumscribed diameter
                approx_diameter = np.sqrt(bbox_width**2 + bbox_height**2)
                if approx_diameter > max_diameter:
                    attempts += 1
                    continue
            
            # If we get here, the shape meets all constraints
            shapes.append(shape)
            constraint_info = []
            if min_area is not None:
                constraint_info.append(f"area>={min_area:.6e}")
            if max_area is not None:
                constraint_info.append(f"area<={max_area:.6e}")
            if min_perimeter is not None:
                constraint_info.append(f"perimeter>={min_perimeter:.6e}")
            if max_perimeter is not None:
                constraint_info.append(f"perimeter<={max_perimeter:.6e}")
            if min_perimeter_area_ratio is not None:
                perimeter_area_ratio = shape.length / shape.area if shape.area > 0 else float('inf')
                constraint_info.append(f"P/A>={min_perimeter_area_ratio:.1f}")
            if max_perimeter_area_ratio is not None:
                perimeter_area_ratio = shape.length / shape.area if shape.area > 0 else float('inf')
                constraint_info.append(f"P/A<={max_perimeter_area_ratio:.1f}")
            if max_diameter is not None:
                minx, miny, maxx, maxy = shape.bounds
                bbox_width = maxx - minx
                bbox_height = maxy - miny
                approx_diameter = np.sqrt(bbox_width**2 + bbox_height**2)
                constraint_info.append(f"diameter<={max_diameter:.6e}")
            
            # Calculate actual metrics for display
            perimeter_area_ratio = shape.length / shape.area if shape.area > 0 else float('inf')
            
            constraint_str = f" ({', '.join(constraint_info)})" if constraint_info else ""
            print(f"Generated shape {i+1}/{num_shapes}: area={shape.area:.6e}, perimeter={shape.length:.6e}, P/A={perimeter_area_ratio:.1f}{constraint_str}")
            break
        else:
            print(f"Failed to generate shape {i+1}/{num_shapes} meeting constraints after {max_attempts} attempts")
    
    return shapes

def min_enclosing_circle(points):
    # This is a naive O(n^3) implementation, sufficient for small n
    from itertools import combinations
    def circle_from(p1, p2, p3=None):
        if p3 is None:
            # Circle from two points: center is midpoint, radius is half distance
            center = (p1 + p2) / 2
            radius = np.linalg.norm(p1 - p2) / 2
            return center, radius
        else:
            # Circle from three points
            A = p2 - p1
            B = p3 - p1
            C = np.cross(A, B)
            if abs(C) < 1e-12:
                return None, np.inf  # Colinear
            D = (np.dot(A, A) * np.cross(B, [0,0,1]) - np.dot(B, B) * np.cross(A, [0,0,1])) / (2 * C)
            center = p1[:2] + D[:2]
            radius = np.linalg.norm(center - p1[:2])
            return center, radius

    n = len(points)
    # Try all pairs
    min_circle = (None, np.inf)
    for i in range(n):
        for j in range(i+1, n):
            c, r = circle_from(points[i], points[j])
            if all(np.linalg.norm(points[:, :2] - c, axis=1) <= r + 1e-10):
                if r < min_circle[1]:
                    min_circle = (c, r)
    # Try all triplets
    for i, j, k in combinations(range(n), 3):
        c, r = circle_from(points[i], points[j], points[k])
        if c is not None and all(np.linalg.norm(points[:, :2] - c, axis=1) <= r + 1e-10):
            if r < min_circle[1]:
                min_circle = (c, r)
    return min_circle

def generate_grid_points(shape: Polygon, discretization_length: float, padding: float = None) -> np.ndarray:
    """
    Generate a square grid of points within the given shape.
    
    Args:
        shape: The shape object to generate points within
        discretization_length: The spacing between grid points in meters
        padding: The amount of padding to add to the shape in meters
    Returns:
        Numpy array of all [x, y, is_inside] coordinates in the grid (unfiltered)
    """
    # Get the bounding box of the shape with padding
    vertices = np.array(shape.exterior.coords)
    # min_x = np.min(vertices[:, 0]) - discretization_length
    # min_y = np.min(vertices[:, 1]) - discretization_length
    # max_x = np.max(vertices[:, 0]) + discretization_length
    # max_y = np.max(vertices[:, 1]) + discretization_length

    center, radius = min_enclosing_circle(vertices)
    min_x = center[0] - radius - discretization_length
    min_y = center[1] - radius - discretization_length
    max_x = center[0] + radius + discretization_length
    max_y = center[1] + radius + discretization_length
    
    # Make the grid square by using the larger dimension
    width = max_x - min_x
    height = max_y - min_y
    grid_size = max(width, height)
    
    # Adjust bounds to make square
    if width < height:
        squaring_pad = (height - width) / 2
        max_x += squaring_pad
        min_x -= squaring_pad
    else:
        squaring_pad = (width - height) / 2
        max_y += squaring_pad
        min_y -= squaring_pad
        
    if padding is not None:
        max_x += padding
        min_x -= padding
        max_y += padding
        min_y -= padding
    
    # Calculate number of points (same in both dimensions for square grid)
    num_points = int(grid_size / discretization_length) + 1
    
    all_points = []
    
    # Generate points over the square grid and flag those inside the shape
    for i in range(num_points):
        for j in range(num_points):
            x = min_x + i * discretization_length
            y = min_y + j * discretization_length
            point = Point(x, y)
            is_inside = int(shape.contains(point))
            all_points.append([x, y, is_inside])
    
    return np.array(all_points)

def generate_grid_points_fast(shape: Polygon, discretization_length: float, padding: float = None) -> np.ndarray:
    """
    Generate a square grid of points within the given shape.
    
    Args:
        shape: The shape object to generate points within
        discretization_length: The spacing between grid points in meters
        padding: The amount of padding to add to the shape in meters
    Returns:
        Numpy array of all [x, y, is_inside] coordinates in the grid (unfiltered)
    """
    # Get the bounding box of the shape with padding
    vertices = np.array(shape.exterior.coords)
    # min_x = np.min(vertices[:, 0]) - discretization_length
    # min_y = np.min(vertices[:, 1]) - discretization_length
    # max_x = np.max(vertices[:, 0]) + discretization_length
    # max_y = np.max(vertices[:, 1]) + discretization_length

    center, radius = min_enclosing_circle(vertices)
    min_x = center[0] - radius - discretization_length
    min_y = center[1] - radius - discretization_length
    max_x = center[0] + radius + discretization_length
    max_y = center[1] + radius + discretization_length
    
    # Make the grid square by using the larger dimension
    width = max_x - min_x
    height = max_y - min_y
    grid_size = max(width, height)
    
    # Adjust bounds to make square
    if width < height:
        squaring_pad = (height - width) / 2
        max_x += squaring_pad
        min_x -= squaring_pad
    else:
        squaring_pad = (width - height) / 2
        max_y += squaring_pad
        min_y -= squaring_pad
        
    if padding is not None:
        max_x += padding
        min_x -= padding
        max_y += padding
        min_y -= padding
    
    # Calculate number of points (same in both dimensions for square grid)
    num_points = int(grid_size / discretization_length) + 1

    # Vectorized grid generation
    xs = min_x + np.arange(num_points, dtype=np.float64) * discretization_length
    ys = min_y + np.arange(num_points, dtype=np.float64) * discretization_length
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    # Vectorized point-in-polygon test
    if _shp_vectorized is not None:
        inside_mask = _shp_vectorized.contains(shape, X, Y)
    else:
        # Fallback to prepared geometry with Python loop if vectorized not available
        prepared = prep(shape)
        Xf = X.ravel()
        Yf = Y.ravel()
        inside_list = [int(prepared.contains(Point(x, y))) for x, y in zip(Xf, Yf)]
        inside_mask = np.array(inside_list, dtype=bool).reshape(X.shape)

    # Stack into Nx3 array [x, y, is_inside]
    out = np.column_stack([
        X.ravel(),
        Y.ravel(),
        inside_mask.astype(np.int32).ravel(),
    ])
    return out

# %%

if __name__ == "__main__":
    print("Testing controllable union and subtraction shape counts...")

    # Test 1: Simple shapes - 1 union, 0 subtraction (solid basic shapes)
    print("\n=== Test 1: Single solid shapes (1 union, 0 subtract) ===")
    simple_shapes = generate_shapes(
        num_shapes=10,
        num_union_shapes=None,
        num_subtract_shapes=None,
        hollowness_bias=0.25,
        max_attempts=50,
        # hole=True
    )

    # Combine all shapes for plotting
    all_shapes = simple_shapes

    if all_shapes:
        print(f"\nSuccessfully generated {len(all_shapes)} shapes total")
        plot_all_shapes(all_shapes)
    else:
        print("Failed to generate any shapes")
    

# %%
