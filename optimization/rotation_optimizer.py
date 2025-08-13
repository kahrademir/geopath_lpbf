
# %%
import torch
import numpy as np
import os
import json
import glob
import sys
import importlib
import matplotlib.pyplot as plt
from shapely.geometry import Point
from datetime import datetime

# Add parent directory to sys.path to allow importing modules
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the predict_temperature_field function from use_model.py
from mLearning.use_model import predict_temperature_field

# === CONFIGURATION ===
# Change this line to switch models (e.g., 'cnn_1', 'cnn_2', etc.)
MODEL_MODULE = 'cnn_12'

# Dynamic import based on MODEL_MODULE
model_module = importlib.import_module(f'mLearning.models.{MODEL_MODULE}')
myModel = model_module.myModel

from toolpath.path_gen import generate_toolpath
from geometry.geometry_samples import generate_shapes, plot_all_shapes


class RotationOptimizer:
    """
    Simple rotation optimizer for finding optimal toolpath angles to minimize temperature.
    Uses the predict_temperature_field function from use_model.py for consistency.
    """
    
    def __init__(self, checkpoint_path, pattern="uniraster", force_cpu=False):
        """Initialize the optimizer with a trained model."""
        self.checkpoint_path = checkpoint_path
        if force_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pattern = pattern
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model."""
        # Load config from checkpoint
        config_path = os.path.join(self.checkpoint_path, 'config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get model kwargs
        model_kwargs = config.get('model_kwargs', {'in_channels': 2, 'out_channels': 1})
        
        # Setup model
        self.model = myModel(**model_kwargs)
        # Load state dict and move to device
        state_dict = torch.load(os.path.join(self.checkpoint_path, 'best_model.pth'), map_location='cpu')
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Verify device placement
        test_param = next(self.model.parameters())
        if test_param.device != self.device:
            print(f"WARNING: Model device ({test_param.device}) doesn't match target device ({self.device})")
        else:
            print(f"✓ Model successfully moved to {self.device}")
    
    def predict_temperature(self, shape, angle, all_points=None, sdf_field=None):
        """
        Predict temperature field for a given shape and toolpath angle.
        Uses the predict_temperature_field function from use_model.py.
        
        Args:
            shape: Shapely geometry object
            angle: Toolpath raster angle in degrees
            all_points: Optional pre-computed grid points (for efficiency)
            sdf_field: Optional pre-computed SDF field (for efficiency)
            
        Returns:
            temperature_field: 2D numpy array of predicted temperatures
            fields: List of input fields [sdf, time, grad]
            all_points: Grid points used for prediction
        """
        # Generate toolpath
        toolpath = generate_toolpath(shape, raster_angle=angle, pattern=self.pattern)
        
        # Use the predict_temperature_field function from use_model.py
        predicted_temp, fields, all_points = predict_temperature_field(
            model=self.model,
            polygon=shape,
            toolpath=toolpath,
            raster_angle=angle,
            all_points=all_points,
            sdf_field=sdf_field
        )
        
        return predicted_temp, fields, all_points
    
    def evaluate_objective(self, temperature_field, mask, objective='max_temp'):
        """
        Evaluate optimization objective from temperature field.
        
        Args:
            temperature_field: 2D array of temperatures
            mask: 2D boolean array of valid regions
            objective: Objective type ('max_temp', 'mean_temp', 'temp_std', 'homogeneity')
            
        Returns:
            objective_value: Scalar objective value
        """
        valid_temps = temperature_field[mask]
        
        if len(valid_temps) == 0:
            return float('inf')  # Invalid solution
        
        if objective == 'max_temp':
            return np.max(valid_temps)
        elif objective == 'mean_temp':
            return np.mean(valid_temps)
        elif objective == 'temp_std':
            return np.std(valid_temps)
        elif objective == 'homogeneity':
            return np.std(valid_temps)  # Same as temp_std
        else:
            raise ValueError(f"Unknown objective: {objective}")
    
    def optimize_rotation(self, shape, angle_range=(0, 360), num_angles=36, objective='max_temp'):
        """
        Find optimal rotation angle for a given shape.
        Uses efficient caching to avoid recomputing shape-dependent data.
        
        Args:
            shape: Shapely geometry object
            angle_range: Tuple of (min_angle, max_angle) in degrees
            num_angles: Number of angles to evaluate
            objective: Optimization objective ('max_temp', 'mean_temp', 'temp_std', 'homogeneity')
            
        Returns:
            optimal_angle: Best angle in degrees
            optimal_value: Best objective value
            results: Dictionary with all angles and their objective values
        """
        angles = np.linspace(angle_range[0], angle_range[1], num_angles, endpoint=False)
        objective_values = []
        
        print(f"Optimizing {num_angles} angles from {angle_range[0]}° to {angle_range[1]}°...")
        
        # Get initial prediction to extract grid points and SDF field for caching
        initial_temp, initial_fields, all_points = self.predict_temperature(shape, angles[0])
        sdf_field = initial_fields[0]  # SDF field is the first field
        
        # Create mask from the first prediction
        grid_size = int(np.sqrt(len(all_points)))
        inside_mask = all_points[:, 2].reshape((grid_size, grid_size)).astype(bool)
        
        for i, angle in enumerate(angles):
            # Use cached grid points and SDF field for efficiency
            temperature_field, _, _ = self.predict_temperature(
                shape, angle, all_points=all_points, sdf_field=sdf_field
            )
            obj_value = self.evaluate_objective(temperature_field, inside_mask, objective)
            objective_values.append(obj_value)
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_angles} angles...")
        
        objective_values = np.array(objective_values)
        
        # Find optimal angle (minimize objective)
        optimal_idx = np.argmin(objective_values)
        optimal_angle = angles[optimal_idx]
        optimal_value = objective_values[optimal_idx]
        
        results = {
            'angles': angles,
            'objective_values': objective_values,
            'optimal_angle': optimal_angle,
            'optimal_value': optimal_value,
            'objective': objective
        }
        
        print(f"\nOptimal angle: {optimal_angle:.1f}° with {objective} = {optimal_value:.3f}")
        
        return optimal_angle, optimal_value, results
    
    def save_optimization_results(self, shape, results, base_dir="/mnt/c/Users/jamba/sim_data", dir_name=None):
        """
        Save comprehensive optimization results including temperature fields, toolpaths, and metadata.
        
        Args:
            shape: Shapely geometry object
            results: Dictionary containing optimization results
            base_dir: Base directory for saving results
            dir_name: Optional directory name for saving results. If None, use timestamp.
        """
        # Determine directory name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dir_name is not None:
            optimization_dir = os.path.join(base_dir, dir_name)
        else:            
            optimization_dir = os.path.join(base_dir, f"OPTIMIZATION_{timestamp}")
        
        # Create main optimization directory
        os.makedirs(optimization_dir, exist_ok=True)
        print(f"Saving optimization results to: {optimization_dir}")
        
        # Save metadata
        metadata = {
            "optimization_info": {
                "timestamp": timestamp,
                "objective": results['objective'],
                "angle_range": [float(results['angles'][0]), float(results['angles'][-1])],
                "num_angles": len(results['angles']),
                "optimal_angle": float(results['optimal_angle']),
                "optimal_value": float(results['optimal_value']),
                "pattern": self.pattern
            },
            "model_info": {
                "checkpoint_path": self.checkpoint_path,
                "model_module": MODEL_MODULE,
                "device": str(self.device)
            },
            "shape": shape.wkt if shape is not None else None
        }
        
        metadata_path = os.path.join(optimization_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save results summary
        results_summary = {
            "angles": [float(x) for x in results['angles'].tolist()],
            "objective_values": [float(x) for x in results['objective_values'].tolist()],
            "optimal_angle": float(results['optimal_angle']),
            "optimal_value": float(results['optimal_value'])
        }
        
        summary_path = os.path.join(optimization_dir, "optimization_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Save data for each angle
        print("Saving temperature fields and toolpaths for each angle...")
        for i, angle in enumerate(results['angles']):
            # Create angle-specific directory
            angle_dir = os.path.join(optimization_dir, f"angle_{i:03d}_{angle:.1f}")
            os.makedirs(angle_dir, exist_ok=True)
            
            # Generate toolpath for this angle
            toolpath = generate_toolpath(shape, raster_angle=angle, pattern=self.pattern)
            
            # Get temperature field and other data
            temp_field, fields, all_points = self.predict_temperature(shape, angle)
            
            # Save temperature field
            np.save(os.path.join(angle_dir, "temperature_field.npy"), temp_field)
            
            # Save input fields (SDF, time, gradient)
            np.save(os.path.join(angle_dir, "sdf_field.npy"), fields[0])
            np.save(os.path.join(angle_dir, "time_field.npy"), fields[1])
            np.save(os.path.join(angle_dir, "gradient_field.npy"), fields[2])
            
            # Save toolpath
            np.save(os.path.join(angle_dir, "toolpath.npy"), toolpath)
            
            # Save grid points and mask
            np.save(os.path.join(angle_dir, "grid_points.npy"), all_points)
            
            # Save angle-specific metadata
            angle_metadata = {
                "angle": float(angle),
                "angle_index": i,
                "objective_value": float(results['objective_values'][i]),
                "toolpath_stats": {
                    "num_points": len(toolpath),
                    "total_time": float(toolpath[-1, 0]) if len(toolpath) > 0 else 0.0,
                    "pattern": self.pattern
                }
            }
            
            angle_metadata_path = os.path.join(angle_dir, "metadata.json")
            with open(angle_metadata_path, 'w') as f:
                json.dump(angle_metadata, f, indent=2)
            
            if (i + 1) % 10 == 0:
                print(f"Saved data for {i + 1}/{len(results['angles'])} angles...")
        
        print(f"Optimization results saved successfully to: {optimization_dir}")
        return optimization_dir


# %%
def get_shapes_from_JSON(json_file, indices=None):
    """
    Get shape geometries (as WKT) from a JSON file in the format of filtered_shapes.json.
    Args:
        json_file (str): Path to the JSON file (e.g., geometry/pa_control/filtered_shapes.json).
        indices (list of int or None): Indices of shapes to extract (these correspond to the 'index' field in each shape dict).
            If None, all shapes are returned.
    Returns:
        list of shapely.geometry.Polygon: List of Polygon objects for the selected shapes.
    """
    import json
    from shapely import wkt
    with open(json_file, 'r') as f:
        data = json.load(f)
    shapes = data["shapes"]
    if indices is None:
        # Return all shapes
        return [wkt.loads(shape["wkt"]) for shape in shapes]
    else:
        # Build a mapping from the 'index' field to the shape dict
        index_to_shape = {shape["index"]: shape for shape in shapes}
        # Extract shapes in the order of the provided indices
        return [wkt.loads(index_to_shape[i]["wkt"]) for i in indices]


# Example usage
if __name__ == "__main__":
    # Load shape from metadata using WKT
    import json
    from shapely import wkt

    # Use scalar parameters (between 0 and 1) to interpolate between min and max perimeter/area ratio and area
    perimeter_area_ratio_scalar = 0.35  # 0.0 = min, 1.0 = max, can be set as desired
    min_perimeter_area_ratio_Circle = 2e-3
    max_perimeter_area_ratio_Ring = 5e-3
    # Center value for the ratio
    perimeter_area_ratio_center = min_perimeter_area_ratio_Circle + perimeter_area_ratio_scalar * (max_perimeter_area_ratio_Ring - min_perimeter_area_ratio_Circle)
    # Define a small window (e.g., ±0.1e-3) around the center value
    perimeter_area_ratio_window = 0.1e-3
    min_perimeter_area_ratio = perimeter_area_ratio_center - perimeter_area_ratio_window
    max_perimeter_area_ratio = perimeter_area_ratio_center + perimeter_area_ratio_window


    area_center = 3.14*1e-6
    area_window = 0.25e-6  # e.g., ±0.01e-6 around the center value
    min_area = area_center - area_window
    max_area = area_center + area_window

    # Generate a single random shape (can adjust parameters as needed)
    # shape = generate_shapes(
    #     num_shapes=10000,
    #     # hole=True,
    #     # num_union_shapes=1,
    #     # hollowness_bias=0.0,
    #     # min_perimeter_area_ratio=min_perimeter_area_ratio,
    #     # max_perimeter_area_ratio=max_perimeter_area_ratio,
    #     # max_attempts=10000,
    #     min_area=min_area,
    #     max_area=max_area
    # )[0]
    indices = [291, 393, 449]
    shapes = get_shapes_from_JSON(os.path.join(parent_dir, "geometry", "pa_control", "filtered_shapes.json"), indices)
    shape = shapes[0]


    plot_all_shapes([shape])

    # Initialize optimizer (use CUDA if available)
    checkpoint_path = "/home/jamba/research/thesis-v2/mLearning/checkpoints/cnn_12_2025-08-04_23-09-22/"
    optimizer = RotationOptimizer(checkpoint_path, pattern="uniraster", force_cpu=False)

    # Optimize rotation
    optimal_angle, optimal_value, results = optimizer.optimize_rotation(
        shape, num_angles=100, objective='homogeneity'
    )

    # Save optimization results
    optimization_dir = optimizer.save_optimization_results(shape, results, dir_name="OPTIMIZATION_LOW_PA_2")
    print(f"Results saved to: {optimization_dir}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(results['angles'], results['objective_values'])
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Objective Value')
    plt.title('Optimization Results')
    plt.grid(True)
    plt.show()

    # Plot best and worst orientations
    best_angle = results['optimal_angle']
    worst_idx = np.argmax(results['objective_values'])
    worst_angle = results['angles'][worst_idx]

    print(f"Best angle: {best_angle:.1f}° (homogeneity: {results['optimal_value']:.3f})")
    print(f"Worst angle: {worst_angle:.1f}° (homogeneity: {results['objective_values'][worst_idx]:.3f})")
    # %%
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Generate and plot best orientation
    temp_best, _, all_points = optimizer.predict_temperature(shape, best_angle)
    # Create mask for visualization (same logic as in predict_temperature_field)
    grid_size = int(np.sqrt(len(all_points)))
    inside_mask = all_points[:, 2].reshape((grid_size, grid_size)).astype(bool)
    temp_best_masked = np.where(inside_mask, temp_best, np.nan)

    im1 = axes[0].imshow(temp_best_masked, cmap='hot', origin='lower', vmin=1200, vmax=1800)
    axes[0].set_title(f'Best Orientation ({best_angle:.1f}°)\nHomogeneity: {results["optimal_value"]:.3f}')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Generate and plot worst orientation  
    temp_worst, _, all_points = optimizer.predict_temperature(shape, worst_angle)
    temp_worst_masked = np.where(inside_mask, temp_worst, np.nan)

    im2 = axes[1].imshow(temp_worst_masked, cmap='hot', origin='lower', vmin=1200, vmax=1800)
    axes[1].set_title(f'Worst Orientation ({worst_angle:.1f}°)\nHomogeneity: {results["objective_values"][worst_idx]:.3f}')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.show()  
# %%
