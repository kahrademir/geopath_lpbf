import os
import sys
import numpy as np
import torch
from scipy.interpolate import RBFInterpolator
from scipy.ndimage import rotate as rotate_image
# Add parent directory to sys.path to allow importing geometry and toolpath modules
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory (project root) to the path
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from geometry.geometry_samples import generate_grid_points
from toolpath.fields import calculate_signed_distance_field
from solver.config import get_mesh_size_presets

def min_enclosing_circle(points):
    """
    # Computes the minimum enclosing circle for a set of points.
    # This is used to determine the maximum characteristic length of the polygon,
    # ensuring that when the shape is rotated, it can be padded sufficiently to prevent clipping.
    """


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

def preprocess_input_fields(polygon, toolpath, 
                           all_points=None,
                           sdf_field=None):
    """
    Preprocess input fields for temperature prediction.
    
    Args:
        polygon: Shapely Polygon representing the part geometry
        toolpath: Numpy array of shape (n_points, 3) with [time, x, y] coordinates
        all_points: Pre-computed grid points (optional)
        sdf_field: Pre-computed SDF field (optional)
    
    Returns:
        tuple: (raw_fields, inside_mask, all_points, grid_size)
            - raw_fields: List of raw input fields [sdf, time, grad]
            - inside_mask: Boolean mask for points inside the polygon
            - all_points: All grid points with inside/outside information
            - grid_size: Size of the grid (grid_size x grid_size)
    """
    
    # Get mesh size configuration for discretization
    mesh_config = get_mesh_size_presets('fine')
    discretization_length = mesh_config['laser']
    
    # Generate evaluation points (grid points within the shape)
    if all_points is None:
        all_points = generate_grid_points(polygon, discretization_length)
    grid_points = all_points[:, :2]
    grid_size = int(np.sqrt(len(grid_points)))
    
    # Check minimum grid size for U-Net
    if grid_size < 8:
        raise ValueError(f"Grid size {grid_size} is too small for U-Net. Minimum size is 8x8.")
    
    # Calculate SDF field
    if sdf_field is None:
        sdf_values = calculate_signed_distance_field(grid_points, polygon)
        sdf_field = sdf_values.reshape((grid_size, grid_size))
    
    # Calculate time field using RBF interpolation
    toolpath_points = toolpath[:, 1:3]
    toolpath_times = toolpath[:, 0]
    
    # Use RBF interpolation for smoother time field
    rbf_interp = RBFInterpolator(toolpath_points, toolpath_times, kernel='linear')
    time_values = rbf_interp(grid_points)
    time_field = time_values.reshape((grid_size, grid_size))
    
    # Calculate gradient magnitude
    grad_y, grad_x = np.gradient(time_field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Create inside mask
    inside_mask = all_points[:, 2].reshape((grid_size, grid_size))
    inside_mask = inside_mask.astype(bool)
    
    # Prepare raw fields for model input
    raw_fields = [sdf_field, time_field, grad_mag]
    
    return raw_fields, inside_mask, all_points, grid_size


def predict_temperature_field(model, polygon, 
                                toolpath=None, 
                                sdf_max=0.001, 
                                time_max=0.08, 
                                temp_min=1000.0, 
                                temp_max=2000.0,
                                grad_max=0.002,
                                grad_min=0.0,
                                raster_angle=0,
                                input_fields=None,    
                                all_points=None
                                ):
    """
    Predict temperature field using a trained torch model.
    
    Args:
        model: Trained torch model (should be in eval mode)
        polygon: Shapely Polygon representing the part geometry
        toolpath: Numpy array of shape (n_points, 3) with [time, x, y] coordinates
        sdf_max: Maximum SDF value for normalization (default: 0.001)
        time_max: Maximum time value for normalization (default: 0.08)
        temp_min: Minimum temperature value for normalization (default: 1000.0)
        temp_max: Maximum temperature value for normalization (default: 2000.0)
        grad_max: Maximum gradient value for normalization (default: 0.002)
        grad_min: Minimum gradient value for normalization (default: 0.0)
        raster_angle: Raster angle in degrees (default: 0)
        input_fields: Pre-computed input fields [sdf, time, grad] (optional)
        all_points: Pre-computed grid points (optional)
        sdf_field: Pre-computed SDF field (optional)
    Returns:
        numpy.ndarray: Predicted temperature field (denormalized) of shape (grid_size, grid_size)
        numpy.ndarray: Input fields used for prediction [sdf, time, grad] (denormalized)
        numpy.ndarray: All points of shape (N, 3) [inside/outside, x, y]
    """
    
    # Set model to evaluation mode
    model.eval()
    
    # Preprocess input fields (raw calculations only)
    if input_fields is None:
        raw_fields, inside_mask, all_points, _ = preprocess_input_fields(
            polygon, toolpath, all_points, sdf_field
        )
    else:
        raw_fields = input_fields
        grid_size = int(np.sqrt(len(all_points)))
        inside_mask = all_points[:, 2].reshape((grid_size, grid_size))

    # Mask fields to only include inside points
    min_values = [0.0, 0.0, grad_min]  # [sdf_min, time_min, grad_min]
    masked_fields = []
    for field, min_val in zip(raw_fields, min_values):
        masked_field = np.where(inside_mask, field, min_val)
        masked_fields.append(masked_field)
    
    # Normalize fields
    max_values = [sdf_max, time_max, grad_max]  # [sdf_max, time_max, grad_max]
    normalized_fields = []
    for field, min_val, max_val in zip(masked_fields, min_values, max_values):
        normalized_field = (field - min_val) / (max_val - min_val)
        normalized_fields.append(normalized_field)
    
    # === ROTATION ALIGNMENT LOGIC ===
    # Stack all available normalized fields into input tensor (C, H, W)
    input_fields = np.stack(normalized_fields, axis=0)
    
    # Rotate input fields by -raster_angle to align to 0 degrees (model's training condition)
    aligned_input = np.zeros_like(input_fields)
    for c in range(input_fields.shape[0]):
        aligned_input[c] = rotate_image(input_fields[c], -raster_angle, reshape=False, order=1)
    
    # Prepare input tensor for model and move to same device as model
    input_tensor = torch.tensor(aligned_input, dtype=torch.float32).unsqueeze(0)
    
    # Get device from model parameters and move input tensor to same device
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # Run model inference
    with torch.no_grad():
        aligned_prediction = model(input_tensor)
    
    # Convert prediction to numpy
    aligned_prediction = aligned_prediction.squeeze().cpu().numpy()
    
    # Rotate prediction back by +raster_angle to get result at original angle
    predicted_temp = rotate_image(aligned_prediction, raster_angle, reshape=False, order=1)
    
    # Mask predicted temperature to only include inside points
    predicted_temp = np.where(inside_mask, predicted_temp, 0)
    # denormalize temperature
    predicted_temp = predicted_temp * (temp_max - temp_min) + temp_min
    
    # Prepare fields for return (original unnormalized fields)
    fields = [raw_fields[0], raw_fields[1], raw_fields[2]]
    
    return predicted_temp, fields, all_points


if __name__ == "__main__":
    # Example usage
    # Add parent directory to sys.path to allow importing modules
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    
    from models.cnn_12 import myModel
    
    from shapely.geometry import Polygon
    from geometry.geometry_samples import generate_shapes, plot_all_shapes
    from toolpath.path_gen import generate_toolpath
    import matplotlib.pyplot as plt
    # Create a simple square polygon
    # polygon = Polygon([(0, 0), (0.001, 0), (0.003, 0.001), (0, 0.005)])
    polygon = generate_shapes(
        num_shapes=1,
        hollowness_bias=0.25,
        max_attempts=50,
        min_area=3e-6
        # hole=True
    )[0]
    plot_all_shapes([polygon])

    # Generate a simple toolpath
    raster_angle = 0
    toolpath = generate_toolpath(polygon, raster_angle=raster_angle, pattern="uniraster")

    # Load model
    model = myModel(in_channels=2, out_channels=1, bilinear=True, kernel_size=7)
    model_path = "/home/jamba/research/thesis-v2/mLearning/checkpoints/cnn_12_2025-08-02_21-07-40/best_model.pth"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Make prediction
    temps = []
    predicted_temp, masked_fields, all_points = predict_temperature_field(model, polygon, toolpath, raster_angle=raster_angle)
    temps.append(predicted_temp)
    for i in np.linspace(0, 360, 16)[1:]:
        toolpath = generate_toolpath(polygon, raster_angle=i, pattern="uniraster")
        predicted_temp, masked_fields, _ = predict_temperature_field(model, polygon, toolpath, raster_angle=i, all_points=all_points, sdf_field=masked_fields[0])
        temps.append(predicted_temp)

    # Simple animator for the predicted temperature fields
    import matplotlib.animation as animation

    fig, ax = plt.subplots(figsize=(10, 10))
    ims = []
    for temp in temps:
        im = ax.imshow(
            temp.squeeze(),
            cmap='hot',
            origin='lower',
            extent=[all_points[0].min(), all_points[0].max(), all_points[1].min(), all_points[1].max()],
            animated=True,
            vmin=1300,
            vmax=1800
        )
        ims.append([im])

    # ax.set_title("Predicted Max Temperature Field (Rotation Animation)")
    plt.colorbar(ims[0][0], ax=ax, label="Temperature (K)")
    animation_time = 4 # seconds
    ani = animation.ArtistAnimation(fig, ims, interval=animation_time*1000/len(temps), blit=True)
    plt.show()