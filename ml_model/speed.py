# %%
import numpy as np
import sys
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


sys.path.append('/home/jamba/research/thesis-v2/')
from toolpath.path_gen import generate_toolpath
from solver.preprocess import preprocess_input_fields
from toolpath.path_gen import generate_toolpath_f
# %%
# 1. Make a circle (as a polygon approximation)
def make_circle(center=(0.0, 0.0), radius=0.001, num_points=100):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    coords = np.stack([x, y], axis=1)
    return Polygon(coords)

shape = make_circle(center=(0.0, 0.0), radius=0.001, num_points=50)
L = 2e-3

# Define a constant speed function
def constant_speed(*args, **kwargs):
    return 0.5  # m/s

def spatial_oscillating_speed(
    t, x=None, y=None, base_speed=1.0, amplitude=0.5,
    wavelength=1e-4
):
    """
    Returns a speed that oscillates sinusoidally over space (x, y),
    using characteristic spatial wavelengths.
    Args:
        t: Time (seconds)
        x, y: Spatial coordinates (meters)
        base_speed: The average speed (m/s)
        amplitude: The amplitude of oscillation (m/s)
        wavelength: Wavelength of spatial oscillation (meters)
    Returns:
        Speed at (t, x, y) (m/s), always >= 0.01
    """
    if amplitude >= base_speed:
        raise ValueError("Amplitude must be less than the base_speed to ensure speed is always non-negative.")

    osc = np.sin(2 * np.pi * y / wavelength)
    # osc is now in [-1, 1], so speed varies between (base_speed - amplitude) and (base_speed + amplitude)
    speed = base_speed + amplitude * osc
    # Ensure speed is always positive and not too small
    return speed



# Set parameters for the test
discretization_distance = 2e-5  # meters


# Generate fields for a list of oscillating periods
wavelengths = [100, L/1.5, L/4, L/8]  # m, example frequencies to test
fields_by_wavelength = {}

# Precompute all_points and sdf_field once, then reuse in subsequent iterations
all_points = None
sdf_field = None

# Generate fields and predictions for each wavelength, then plot everything
num_speeds = len(wavelengths)
fig, axes = plt.subplots(num_speeds, 3, figsize=(15, 5 * num_speeds))

# Import the model prediction function
from mLearning.use_model import predict_temperature_field
from mLearning.models.cnn_16 import myModel
import torch

model = myModel()
model.load_state_dict(torch.load('/home/jamba/research/thesis-v2/mLearning/checkpoints/cnn_16_2025-08-05_02-20-31/best_model.pth'))
model.eval()

for i, wavelength in enumerate(wavelengths):
    # Generate the toolpath with the current oscillating frequency
    toolpath = generate_toolpath_f(
        shape=shape,
        speed_func=lambda t, x, y, wavelength=wavelength: spatial_oscillating_speed(t, x, y, base_speed=1.0, amplitude=0.7, wavelength=wavelength),
        discretization_distance=discretization_distance,
    )
    print(f"Generated toolpath shape for wavelength {wavelength} m:", toolpath.shape)

    # On first iteration, let preprocess_input_fields compute all_points and sdf_field
    # On subsequent iterations, reuse them
    raw_fields, inside_mask, all_points, grid_size = preprocess_input_fields(
        shape,
        toolpath,
        all_points=all_points,
        sdf_field=sdf_field,
        discretization_length=discretization_distance
    )
    # Save sdf_field after first computation
    if i == 0:
        sdf_field = raw_fields[0]

    # Store the fields for later analysis
    fields_by_wavelength[wavelength] = {
        "toolpath": toolpath,
        "raw_fields": raw_fields,
        "inside_mask": inside_mask,
        "all_points": all_points,
        "grid_size": grid_size
    }

    # Generate speed field visualization
    speed_field = np.zeros((grid_size, grid_size))
    for row in range(grid_size):
        for col in range(grid_size):
            if inside_mask[row, col]:
                idx = row * grid_size + col
                x, y = all_points[idx, 0], all_points[idx, 1]
                speed_field[row, col] = spatial_oscillating_speed(0, x, y, base_speed=1.0, amplitude=0.5, wavelength=wavelength)
    
    # Mask the speed field
    masked_speed_field = np.where(inside_mask, speed_field, np.nan)

    # Get gradient field
    grad_field = raw_fields[2]
    # Normalize gradient field individually to [0, 1], ignoring NaNs and only inside the mask
    masked_grad = np.where(inside_mask, grad_field, np.nan)
    if np.any(inside_mask):
        local_min = np.nanmin(masked_grad)
        local_max = np.nanmax(masked_grad)
    else:
        local_min = 0.0
        local_max = 1.0
    if local_max > local_min:
        norm_grad_field = (grad_field - local_min) / (local_max - local_min)
    else:
        norm_grad_field = np.zeros_like(grad_field)
    norm_grad_field_masked = np.where(inside_mask, norm_grad_field, np.nan)

    # Predict temperature
    predicted_temp, fields, all_points = predict_temperature_field(
        model, shape, toolpath, input_fields=raw_fields, all_points=all_points
    )
    masked_pred_temp = np.where(inside_mask, predicted_temp, np.nan)

    # Plot speed field
    im1 = axes[i, 0].imshow(masked_speed_field, cmap='viridis', vmin=0.5, vmax=1.5)
    axes[i, 0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[i, 0], orientation='horizontal', pad=0.1, aspect=30)
    cbar1.set_ticks([0.5, 1.5])
    cbar1.set_label('Speed (m/s)')

    # Plot gradient field
    im2 = axes[i, 1].imshow(norm_grad_field_masked, cmap='magma', vmin=0, vmax=1)
    axes[i, 1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[i, 1], orientation='horizontal', pad=0.1, aspect=30)
    cbar2.set_ticks([0, 1])
    cbar2.set_label('Gradient Field')

    # Plot predicted temperature
    im3 = axes[i, 2].imshow(masked_pred_temp, cmap='hot', vmin=1500, vmax=1800)
    axes[i, 2].axis('off')
    cbar3 = plt.colorbar(im3, ax=axes[i, 2], orientation='horizontal', pad=0.1, aspect=30)
    cbar3.set_ticks([1500, 1800])
    cbar3.set_label('Temperature (K)')

plt.tight_layout()
plt.show()

# %%
# Load and process Cal Bear SVG file
from svgpathtools import svg2paths
from shapely.geometry import Polygon
from shapely.affinity import translate, scale
import matplotlib.pyplot as plt

# Load SVG and convert to scaled polygon
paths, _ = svg2paths('/home/jamba/research/thesis-v2/speed_and_scale/cal_bear_shape.svg')
if paths:
    # Convert first path to polygon coordinates
    main_path = paths[0]
    t_values = np.linspace(0, 1, 300)
    points = [main_path.point(t) for t in t_values]
    coords = [(point.real, point.imag) for point in points]
    cal_bear_polygon = Polygon(coords)
    
    # Fix invalid geometry if needed
    if not cal_bear_polygon.is_valid:
        cal_bear_polygon = cal_bear_polygon.buffer(0)
    
    # Scale to LPBF manufacturing size (2mm width)
    minx, miny, maxx, maxy = cal_bear_polygon.bounds
    scale_factor = 2e-3 / (maxx - minx)
    centroid_x, centroid_y = cal_bear_polygon.centroid.x, cal_bear_polygon.centroid.y
    
    cal_bear_polygon = translate(cal_bear_polygon, xoff=-centroid_x, yoff=-centroid_y)
    cal_bear_polygon = scale(cal_bear_polygon, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
    
    # Add 180 degree rotation
    from shapely.affinity import rotate
    cal_bear_polygon = rotate(cal_bear_polygon, 180, origin=(0, 0))
    # Plot the Cal Bear shape
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x, y = cal_bear_polygon.exterior.xy
    ax.plot(x, y, 'b-', linewidth=2, label='Cal Bear Shape')
    ax.fill(x, y, alpha=0.3, color='blue')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Cal Bear Shape (Scaled for LPBF Manufacturing)')
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    plt.tight_layout()
    plt.show()
    
    print(f"Cal Bear polygon created and scaled by factor {scale_factor:.6f}")
    print(f"Final dimensions: {cal_bear_polygon.bounds}")

# %%
# Generate toolpath and predict temperature for Cal Bear shape
if 'cal_bear_polygon' in locals() and cal_bear_polygon is not None:
    # Generate toolpath with constant speed
    toolpath = generate_toolpath_f(
        shape=cal_bear_polygon,
        speed_func=constant_speed,
        discretization_distance=discretization_distance,
    )
    
    # Preprocess input fields
    raw_fields, inside_mask, all_points, grid_size = preprocess_input_fields(
        cal_bear_polygon,
        toolpath,
        discretization_length=discretization_distance
    )
    
    # Load model and predict temperature
    from mLearning.use_model import predict_temperature_field
    from mLearning.models.cnn_16 import myModel
    import torch
    
    model = myModel()
    model.load_state_dict(torch.load('/home/jamba/research/thesis-v2/mLearning/checkpoints/cnn_16_2025-08-05_02-20-31/best_model.pth'))
    model.eval()
    
    predicted_temp, fields, all_points = predict_temperature_field(
        model, cal_bear_polygon, toolpath, input_fields=raw_fields, all_points=all_points
    )
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Apply mask to all fields for visualization
    masked_sdf = np.where(inside_mask, raw_fields[0], np.nan)
    masked_time = np.where(inside_mask, raw_fields[1], np.nan)
    masked_grad = np.where(inside_mask, raw_fields[2], np.nan)
    masked_temp = np.where(inside_mask, predicted_temp, np.nan)
    
    # SDF field
    im0 = axes[0].imshow(masked_sdf, cmap='copper')
    axes[0].axis('off')
    axes[0].set_title('Signed Distance Field')
    cbar0 = plt.colorbar(im0, ax=axes[0], orientation='horizontal', pad=0.1)
    cbar0.set_label('SDF (m)')
    
    # Time field
    im1 = axes[1].imshow(masked_time, cmap='viridis')
    axes[1].axis('off')
    axes[1].set_title('Time Field')
    cbar1 = plt.colorbar(im1, ax=axes[1], orientation='horizontal', pad=0.1)
    cbar1.set_label('Time (s)')
    
    # Gradient magnitude field
    im2 = axes[2].imshow(masked_grad, cmap='magma')
    axes[2].axis('off')
    axes[2].set_title('Gradient Magnitude')
    cbar2 = plt.colorbar(im2, ax=axes[2], orientation='horizontal', pad=0.1)
    cbar2.set_label('|∇t| (1/m)')
    
    # Predicted temperature field
    im3 = axes[3].imshow(masked_temp, cmap='hot', vmin=1400, vmax=1800)
    axes[3].axis('off')
    axes[3].set_title('Predicted Max Temperature')
    cbar3 = plt.colorbar(im3, ax=axes[3], orientation='horizontal', pad=0.1)
    cbar3.set_label('Temperature (K)')
    
    plt.suptitle('Cal Bear Shape: Input Fields and Temperature Prediction', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\nPrediction Statistics:")
    print(f"Min temperature: {np.nanmin(masked_temp):.1f} K")
    print(f"Max temperature: {np.nanmax(masked_temp):.1f} K")
    print(f"Mean temperature: {np.nanmean(masked_temp):.1f} K")
    print(f"Temperature range: {np.nanmax(masked_temp) - np.nanmin(masked_temp):.1f} K")

# %%
# Save the figure as high-quality SVG
fig.savefig('/home/jamba/research/thesis-v2/speed_and_scale/cal_bear_temperature_prediction.svg', 
            format='svg', dpi=300, bbox_inches='tight')
