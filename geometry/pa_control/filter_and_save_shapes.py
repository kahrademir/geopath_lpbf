#!/usr/bin/env python3
"""
Script to filter shapes within a specific area band, save data to JSON, and plot extreme shapes.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append("/home/jamba/research/thesis-v2/")
from geometry.geometry_samples import generate_shapes, plot_shape

import matplotlib as mpl
mpl.rcParams['font.size'] = 18
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['legend.fontsize'] = 18
mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18


def filter_and_save_shapes(num_shapes=500, min_area=3e-6, max_area=3.3e-6, 
                          output_file="filtered_shapes.json", max_attempts=1000):
    """
    Generate shapes, filter by area band, and save data to JSON.
    
    Args:
        num_shapes: Number of shapes to generate
        min_area: Minimum area in m^2 (3 mm^2 = 3e-6 m^2)
        max_area: Maximum area in m^2 (3.3 mm^2 = 3.3e-6 m^2)
        output_file: JSON file to save the data
        max_attempts: Maximum attempts per shape
    """
    
    print(f"Generating {num_shapes} shapes with area between {min_area*1e6:.1f} and {max_area*1e6:.1f} mm²...")
    
    # Generate shapes with area constraints
    shapes = generate_shapes(
        num_shapes=num_shapes,
        min_area=min_area,
        max_area=max_area,
        max_attempts=max_attempts
    )
    
    print(f"Successfully generated {len(shapes)} shapes within area constraints")
    
    # Calculate metrics for each shape
    shape_data = []
    
    for i, shape in enumerate(shapes):
        if shape.area > 0:  # Avoid division by zero
            perimeter = shape.length
            area = shape.area
            perimeter_area_ratio = perimeter / area
            
            shape_info = {
                "index": i,
                "area_mm2": area * 1e6,  # Convert to mm²
                "perimeter_mm": perimeter * 1e3,  # Convert to mm
                "perimeter_area_ratio": perimeter_area_ratio,
                "area_m2": area,
                "perimeter_m": perimeter,
                "wkt": shape.wkt  # Well-Known Text format
            }
            shape_data.append(shape_info)
    
    # Sort by perimeter-to-area ratio
    shape_data.sort(key=lambda x: x["perimeter_area_ratio"])
    
    # Save to JSON
    output_data = {
        "metadata": {
            "num_shapes": len(shape_data),
            "min_area_mm2": min_area * 1e6,
            "max_area_mm2": max_area * 1e6,
            "min_perimeter_area_ratio": shape_data[0]["perimeter_area_ratio"] if shape_data else None,
            "max_perimeter_area_ratio": shape_data[-1]["perimeter_area_ratio"] if shape_data else None,
            "mean_perimeter_area_ratio": np.mean([s["perimeter_area_ratio"] for s in shape_data]) if shape_data else None,
            "median_perimeter_area_ratio": np.median([s["perimeter_area_ratio"] for s in shape_data]) if shape_data else None
        },
        "shapes": shape_data
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Saved {len(shape_data)} shapes to {output_file}")
    
    # Print statistics
    ratios = [s["perimeter_area_ratio"] for s in shape_data]
    areas_mm2 = [s["area_mm2"] for s in shape_data]
    perimeters_mm = [s["perimeter_mm"] for s in shape_data]
    
    print(f"\nStatistics for filtered shapes:")
    print(f"  Number of shapes: {len(shape_data)}")
    print(f"  Area range: {min(areas_mm2):.3f} - {max(areas_mm2):.3f} mm²")
    print(f"  Perimeter range: {min(perimeters_mm):.3f} - {max(perimeters_mm):.3f} mm")
    print(f"  Perimeter-to-area ratio range: {min(ratios):.2f} - {max(ratios):.2f}")
    print(f"  Mean perimeter-to-area ratio: {np.mean(ratios):.2f}")
    print(f"  Median perimeter-to-area ratio: {np.median(ratios):.2f}")
    
    return shapes, shape_data

def plot_extreme_shapes(shapes, shape_data, save_plots=True):
    """
    Plot the shapes with the highest and lowest perimeter-to-area ratios.
    
    Args:
        shapes: List of shape objects
        shape_data: List of shape data dictionaries
        save_plots: Whether to save the plots
    """
    
    if not shape_data:
        print("No shape data to plot")
        return
    
    # Find shapes with extreme ratios
    min_ratio_shape = shape_data[0]  # Lowest ratio (first after sorting)
    max_ratio_shape = shape_data[-1]  # Highest ratio (last after sorting)
    
    # Get the corresponding shape objects
    min_ratio_index = min_ratio_shape["index"]
    max_ratio_index = max_ratio_shape["index"]
    
    min_ratio_shape_obj = shapes[min_ratio_index]
    max_ratio_shape_obj = shapes[max_ratio_index]
    
    # Create the plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot shape with lowest perimeter-to-area ratio
    axes[0].set_aspect('equal')
    x, y = min_ratio_shape_obj.exterior.xy
    axes[0].plot(x, y, 'b-', linewidth=2)
    
    # Plot interior holes if any
    for interior in min_ratio_shape_obj.interiors:
        x, y = interior.xy
        axes[0].plot(x, y, 'r-', linewidth=1)
    
    axes[0].set_title(f'Lowest P/A Ratio: {min_ratio_shape["perimeter_area_ratio"]:.2f}\n'
                      f'Area: {min_ratio_shape["area_mm2"]:.3f} mm², '
                      f'Perimeter: {min_ratio_shape["perimeter_mm"]:.3f} mm')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('X (mm)')
    axes[0].set_ylabel('Y (mm)')
    
    # Plot shape with highest perimeter-to-area ratio
    axes[1].set_aspect('equal')
    x, y = max_ratio_shape_obj.exterior.xy
    axes[1].plot(x, y, 'b-', linewidth=2)
    
    # Plot interior holes if any
    for interior in max_ratio_shape_obj.interiors:
        x, y = interior.xy
        axes[1].plot(x, y, 'r-', linewidth=1)
    
    axes[1].set_title(f'Highest P/A Ratio: {max_ratio_shape["perimeter_area_ratio"]:.2f}\n'
                      f'Area: {max_ratio_shape["area_mm2"]:.3f} mm², '
                      f'Perimeter: {max_ratio_shape["perimeter_mm"]:.3f} mm')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlabel('X (mm)')
    axes[1].set_ylabel('Y (mm)')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('extreme_shapes.png', dpi=300, bbox_inches='tight')
        print("Saved extreme shapes plot to 'extreme_shapes.png'")
    
    plt.show()
    
    # Print details about the extreme shapes
    print(f"\nShape with LOWEST perimeter-to-area ratio:")
    print(f"  Index: {min_ratio_index}")
    print(f"  Perimeter-to-area ratio: {min_ratio_shape['perimeter_area_ratio']:.2f}")
    print(f"  Area: {min_ratio_shape['area_mm2']:.3f} mm²")
    print(f"  Perimeter: {min_ratio_shape['perimeter_mm']:.3f} mm")
    
    print(f"\nShape with HIGHEST perimeter-to-area ratio:")
    print(f"  Index: {max_ratio_index}")
    print(f"  Perimeter-to-area ratio: {max_ratio_shape['perimeter_area_ratio']:.2f}")
    print(f"  Area: {max_ratio_shape['area_mm2']:.3f} mm²")
    print(f"  Perimeter: {max_ratio_shape['perimeter_mm']:.3f} mm")

def create_histogram_from_json(json_file):
    """
    Create a histogram from the saved JSON data.
    
    Args:
        json_file: Path to the JSON file with shape data
    """
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    ratios = [shape["perimeter_area_ratio"] for shape in data["shapes"]]
    
    plt.figure(figsize=(10, 5))
    plt.hist(ratios, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Perimeter-to-Area Ratio')
    plt.ylabel('Frequency')
    # plt.title(f'Histogram of Perimeter-to-Area Ratios')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for mean and median
    mean_ratio = sum(ratios) / len(ratios)
    # median_ratio = data["metadata"]["median_perimeter_area_ratio"]
    plt.axvline(mean_ratio, color='red', linestyle='--', label=f'Mean: {mean_ratio:.2f}')
    # plt.axvline(median_ratio, color='green', linestyle='--', label=f'Median: {median_ratio:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('filtered_shapes_histogram.png', dpi=300, bbox_inches='tight')
    print("Saved histogram to 'filtered_shapes_histogram.png'")
    plt.show()

def plot_grid_of_shapes(file):
    # Load saved JSON and plot 3x5 grid: 5 P/A ratio classes, 3 shapes per class
    from shapely import wkt
    import numpy as np

    with open(file, "r") as f:
        data = json.load(f)
    shapes_sorted = sorted(data["shapes"], key=lambda x: x["perimeter_area_ratio"])
    num_shapes = len(shapes_sorted)
    num_classes = 5
    shapes_per_class = 3

    # Compute class boundaries (split into 5 equal bins by P/A ratio)
    pa_ratios = [shape["perimeter_area_ratio"] for shape in shapes_sorted]
    class_edges = np.linspace(min(pa_ratios), max(pa_ratios), num_classes + 1)

    # For each class, select 3 shapes (evenly spaced within the class, or closest to quantiles)
    selected_shapes_grid = []
    indices_grid = []  # To store indices for printing
    for i in range(num_classes):
        class_min = class_edges[i]
        class_max = class_edges[i + 1]
        # Get all shapes in this class
        class_shapes = [s for s in shapes_sorted if class_min <= s["perimeter_area_ratio"] <= class_max]
        if len(class_shapes) == 0:
            # If no shapes in this class, fill with None
            selected_shapes_grid.append([None] * shapes_per_class)
            indices_grid.append([None] * shapes_per_class)
            continue
        # If fewer than needed, repeat last
        if len(class_shapes) < shapes_per_class:
            selected = class_shapes + [class_shapes[-1]] * (shapes_per_class - len(class_shapes))
            selected_indices = [s["index"] for s in class_shapes] + [class_shapes[-1]["index"]] * (shapes_per_class - len(class_shapes))
        else:
            # Evenly spaced indices within the class
            indices = [int(round(j * (len(class_shapes) - 1) / (shapes_per_class - 1))) for j in range(shapes_per_class)]
            selected = [class_shapes[idx] for idx in indices]
            selected_indices = [class_shapes[idx]["index"] for idx in indices]
        selected_shapes_grid.append(selected)
        indices_grid.append(selected_indices)

    # Print out indices of each shape in the grid
    print("Indices of shapes in the 3x5 grid (rows: shapes per class, columns: P/A classes):")
    for row in range(shapes_per_class):
        row_indices = []
        for col in range(num_classes):
            idx = indices_grid[col][row]
            row_indices.append(str(idx) if idx is not None else "None")
        print(f"Row {row+1}: " + ", ".join(row_indices))

    # Plot 3x5 grid (rows: shapes per class, columns: P/A classes)
    fig, axes = plt.subplots(shapes_per_class, num_classes, figsize=(18, 9))
    for col in range(num_classes):
        for row in range(shapes_per_class):
            ax = axes[row, col]
            shape_info = selected_shapes_grid[col][row]
            if shape_info is None:
                ax.text(0.5, 0.5, 'No shape', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            geom = wkt.loads(shape_info["wkt"])
            if geom.is_empty:
                ax.text(0.5, 0.5, 'Empty', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            else:
                if geom.geom_type == 'Polygon':
                    x, y = geom.exterior.xy
                    ax.fill(x, y, 'gray', alpha=0.7)
                    ax.plot(x, y, 'k-', linewidth=1.5)
                    for interior in geom.interiors:
                        x, y = interior.xy
                        ax.fill(x, y, 'white', alpha=1.0)
                        ax.plot(x, y, 'k-', linewidth=1.5)
                elif geom.geom_type == 'MultiPolygon':
                    for g in geom.geoms:
                        x, y = g.exterior.xy
                        ax.fill(x, y, 'gray', alpha=0.7)
                        ax.plot(x, y, 'k-', linewidth=1.5)
            # Only show title for top row
            if row == 0:
                pa_min = class_edges[col]
                pa_max = class_edges[col + 1]
                ax.set_title(f"P/A: {pa_min:.0f}-{pa_max:.0f}", fontsize=12)
            # Always show area in lower right
            ax.text(0.98, 0.02, f"{shape_info['area_mm2']:.2f} mm²", ha='right', va='bottom', fontsize=9, color='black', transform=ax.transAxes)
            ax.set_aspect('equal')
            ax.axis('off')

    plt.suptitle("3 Examples per Perimeter-to-Area Ratio Class (Low → High)", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("PA_ratio_classes_grid.png", dpi=300, bbox_inches='tight')
    print("Saved 3x5 grid plot of shapes by P/A ratio class as 'PA_ratio_classes_grid.png'")
    plt.show()


def create_flat_PA_distribution(
    input_json="filtered_shapes.json",
    output_json="filtered_shapes_flat_PA.json",
    num_bins=10,
    shapes_per_bin=20,
    random_seed=42
    ):
    """
    Reads a JSON of shapes, computes P/A ratios, and creates a new JSON with a flat distribution of P/A ratios.
    The output JSON will store shapes in WKT format.
    Also plots the resulting P/A ratio distribution.
    Args:
        input_json: Path to input JSON file with shape data.
        output_json: Path to output JSON file for flat-distributed shapes.
        num_bins: Number of bins to divide the P/A ratio range into.
        shapes_per_bin: Number of shapes to select per bin.
        random_seed: Seed for reproducibility.
    """
    import numpy as np
    import json
    from collections import defaultdict
    import matplotlib.pyplot as plt

    np.random.seed(random_seed)
    # Load shapes
    with open(input_json, "r") as f:
        data = json.load(f)
    shapes = data["shapes"]
    if len(shapes) == 0:
        print("No shapes found in input JSON.")
        return

    # Get all P/A ratios
    pa_ratios = np.array([s["perimeter_area_ratio"] for s in shapes])
    min_pa, max_pa = pa_ratios.min(), pa_ratios.max()
    # Bin edges
    bin_edges = np.linspace(min_pa, max_pa, num_bins + 1)
    # Assign each shape to a bin
    bin_indices = np.digitize(pa_ratios, bin_edges, right=False) - 1
    # Fix any that fall on the upper edge
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Collect shapes per bin
    bin_to_shapes = defaultdict(list)
    for idx, shape in enumerate(shapes):
        bin_idx = bin_indices[idx]
        bin_to_shapes[bin_idx].append(shape)

    # For each bin, randomly select up to shapes_per_bin shapes
    selected_shapes = []
    for bin_idx in range(num_bins):
        shapes_in_bin = bin_to_shapes[bin_idx]
        if len(shapes_in_bin) == 0:
            print(f"Warning: No shapes in bin {bin_idx} (P/A {bin_edges[bin_idx]:.2f}-{bin_edges[bin_idx+1]:.2f})")
            continue
        if len(shapes_in_bin) <= shapes_per_bin:
            selected = shapes_in_bin
        else:
            selected = list(np.random.choice(shapes_in_bin, shapes_per_bin, replace=False))
        selected_shapes.extend(selected)

    # Sort selected shapes by P/A ratio for convenience
    selected_shapes.sort(key=lambda s: s["perimeter_area_ratio"])

    # Only keep WKT and essential info in output
    output_shapes = []
    for s in selected_shapes:
        output_shapes.append({
            "index": s.get("index", None),
            "perimeter_area_ratio": s["perimeter_area_ratio"],
            "area_mm2": s["area_mm2"],
            "perimeter_mm": s["perimeter_mm"],
            "wkt": s["wkt"]
        })

    # Prepare output JSON
    output_data = {
        "metadata": {
            "num_bins": num_bins,
            "shapes_per_bin": shapes_per_bin,
            "total_selected_shapes": len(output_shapes),
            "pa_ratio_bin_edges": [float(x) for x in bin_edges],
            "min_pa_ratio": float(min_pa),
            "max_pa_ratio": float(max_pa)
        },
        "shapes": output_shapes
    }
    with open(output_json, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved flat P/A-distributed shapes to {output_json} ({len(output_shapes)} shapes, {num_bins} bins, {shapes_per_bin} per bin, WKT format)")

    # Plot the resulting P/A ratio distribution
    selected_pa_ratios = [s["perimeter_area_ratio"] for s in output_shapes]
    plt.figure(figsize=(8, 5))
    plt.hist(selected_pa_ratios, bins=bin_edges, color='skyblue', edgecolor='black', alpha=0.8, rwidth=0.95)
    plt.xlabel("Perimeter-to-Area Ratio")
    plt.ylabel("Number of Shapes")
    plt.title(f"Histogram of Selected Shapes' P/A Ratios (Flat Distribution Target)\n({num_bins} bins, {shapes_per_bin} per bin)")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Example usage:
# create_flat_PA_distribution("filtered_shapes.json", "filtered_shapes_flat_PA.json", num_bins=10, shapes_per_bin=20)

if __name__ == "__main__":
    # filter_and_save_shapes(num_shapes=2000, output_file="filtered_shapes_for_figure.json")
    # create_flat_PA_distribution(input_json="filtered_shapes_for_figure.json", output_json="filtered_shapes_for_figure_flat_PA.json", num_bins=10, shapes_per_bin=50)
    create_histogram_from_json("/home/jamba/research/thesis-v2/geometry/pa_control/filtered_shapes_for_figure_flat_PA.json")
    # plot_grid_of_shapes("filtered_shapes_for_figure_flat_PA.json")