#!/usr/bin/env python3
"""
Script to generate 10,000 shapes and plot a histogram of their perimeter-to-area ratios.
"""

import numpy as np
import matplotlib.pyplot as plt
from geometry.geometry_samples import generate_shapes

def generate_and_analyze_shapes(num_shapes=10000, min_area=None, max_area=None, 
                               min_perimeter=None, max_perimeter=None,
                               min_perimeter_area_ratio=None, max_perimeter_area_ratio=None,
                               hollowness_bias=0.5, num_union_shapes=None, num_subtract_shapes=None):
    """
    Generate shapes and analyze their perimeter-to-area ratios.
    
    Args:
        num_shapes: Number of shapes to generate
        min_area: Minimum area constraint
        max_area: Maximum area constraint
        min_perimeter: Minimum perimeter constraint
        max_perimeter: Maximum perimeter constraint
        min_perimeter_area_ratio: Minimum perimeter/area ratio constraint
        max_perimeter_area_ratio: Maximum perimeter/area ratio constraint
        hollowness_bias: Bias for hollow vs solid shapes
        num_union_shapes: Number of shapes to union
        num_subtract_shapes: Number of shapes to subtract
    """
    
    print(f"Generating {num_shapes} shapes...")
    
    # Generate shapes
    shapes = generate_shapes(
        num_shapes=num_shapes,
        min_area=min_area,
        max_area=max_area,
        min_perimeter=min_perimeter,
        max_perimeter=max_perimeter,
        min_perimeter_area_ratio=min_perimeter_area_ratio,
        max_perimeter_area_ratio=max_perimeter_area_ratio,
        hollowness_bias=hollowness_bias,
        num_union_shapes=num_union_shapes,
        num_subtract_shapes=num_subtract_shapes,
        max_attempts=1000  # Increase attempts for better constraint satisfaction
    )
    
    print(f"Successfully generated {len(shapes)} shapes")
    
    # Calculate perimeter-to-area ratios
    perimeter_area_ratios = []
    areas = []
    perimeters = []
    
    for shape in shapes:
        if shape.area > 0:  # Avoid division by zero
            ratio = shape.length / shape.area
            perimeter_area_ratios.append(ratio)
            areas.append(shape.area)
            perimeters.append(shape.length)
    
    # Convert to numpy arrays for analysis
    ratios = np.array(perimeter_area_ratios)
    areas = np.array(areas)
    perimeters = np.array(perimeters)
    
    # Print statistics
    print(f"\nPerimeter-to-Area Ratio Statistics:")
    print(f"  Mean: {np.mean(ratios):.2f}")
    print(f"  Median: {np.median(ratios):.2f}")
    print(f"  Std Dev: {np.std(ratios):.2f}")
    print(f"  Min: {np.min(ratios):.2f}")
    print(f"  Max: {np.max(ratios):.2f}")
    print(f"  Q1: {np.percentile(ratios, 25):.2f}")
    print(f"  Q3: {np.percentile(ratios, 75):.2f}")
    
    print(f"\nArea Statistics:")
    print(f"  Mean: {np.mean(areas):.6e}")
    print(f"  Median: {np.median(areas):.6e}")
    print(f"  Min: {np.min(areas):.6e}")
    print(f"  Max: {np.max(areas):.6e}")
    
    print(f"\nPerimeter Statistics:")
    print(f"  Mean: {np.mean(perimeters):.6e}")
    print(f"  Median: {np.median(perimeters):.6e}")
    print(f"  Min: {np.min(perimeters):.6e}")
    print(f"  Max: {np.max(perimeters):.6e}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram of perimeter-to-area ratios
    axes[0, 0].hist(ratios, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('Perimeter-to-Area Ratio')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Histogram of Perimeter-to-Area Ratios')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_ratio = np.mean(ratios)
    median_ratio = np.median(ratios)
    axes[0, 0].axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
    axes[0, 0].axvline(median_ratio, color='green', linestyle='--', label=f'Median: {median_ratio:.2f}')
    axes[0, 0].legend()
    
    # Histogram of areas
    axes[0, 1].hist(areas, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Area')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Histogram of Areas')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Histogram of perimeters
    axes[1, 0].hist(perimeters, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_xlabel('Perimeter')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Perimeters')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot: Area vs Perimeter
    axes[1, 1].scatter(areas, perimeters, alpha=0.5, s=1)
    axes[1, 1].set_xlabel('Area')
    axes[1, 1].set_ylabel('Perimeter')
    axes[1, 1].set_title('Area vs Perimeter')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = np.corrcoef(areas, perimeters)[0, 1]
    axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=axes[1, 1].transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return shapes, ratios, areas, perimeters

if __name__ == "__main__":
    # Example usage with different constraints
    print("Generating shapes with default parameters...")
    
    # Generate shapes with default parameters
    shapes, ratios, areas, perimeters = generate_and_analyze_shapes(
        num_shapes=10000,
        # Uncomment and modify these parameters as needed:
        # min_area=1e-6,
        # max_area=1e-4,
        # min_perimeter=1e-3,
        # max_perimeter=1e-2,
        # min_perimeter_area_ratio=10,
        # max_perimeter_area_ratio=100,
        # hollowness_bias=0.5,
        # num_union_shapes=3,
        # num_subtract_shapes=1
    )
    
    print(f"\nScript completed successfully!")
    print(f"Generated {len(shapes)} shapes")
    print(f"Analyzed {len(ratios)} valid perimeter-to-area ratios")