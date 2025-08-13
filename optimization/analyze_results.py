# %%
import os
import json
import numpy as np

def load_optimization_results(predicted_dir, simulated_dir):
    """
    Load predicted and simulated temperature fields from separate directories for comparison.

    Args:
        predicted_dir (str): Path to the predicted temperature data directory.
        simulated_dir (str): Path to the simulated temperature data directory.

    Returns:
        results (dict): Dictionary containing:
            - predicted_temps: list of np.ndarray (predicted temperature fields)
            - simulated_temps: list of np.ndarray (simulated temperature fields)
            - predicted_angles: list of angles for predicted data
            - simulated_angles: list of angles for simulated data
            - predicted_masks: list of masks for predicted data
            - simulated_masks: list of masks for simulated data
    """

    # Load predicted temperature fields
    predicted_temps = []
    predicted_angles = []
    predicted_masks = []

    # Find all directories in the predicted directory
    pred_angle_dirs = [d for d in os.listdir(predicted_dir) if os.path.isdir(os.path.join(predicted_dir, d))]
    # print(pred_angle_dirs)
    for d in pred_angle_dirs:
        angle_dir = os.path.join(predicted_dir, d)
        
        # Load predicted temperature
        pred_temp_path = os.path.join(angle_dir, "temperature_field.npy")

        if not os.path.exists(pred_temp_path):
            continue  # Skip directories that don't have the expected file
        predicted_temp = np.load(pred_temp_path)
        size = int(np.sqrt(len(predicted_temp)))
        predicted_temps.append(predicted_temp)

        # Load mask for predicted data
        mask_path = os.path.join(angle_dir, "grid_points.npy") # grid points is actually not the correct name for this file. It should be all_points.npy but it was named grid_points.npy by mistake.
        if not os.path.exists(mask_path):
            continue  # Skip directories that don't have the expected file
        mask = np.load(mask_path)
        mask = mask.astype(bool)
        size = int(np.sqrt(len(mask)))
        mask = mask[:,2].reshape(size, size)
        predicted_masks.append(mask)

        # Load metadata for predicted data - use "metadata.json"
        meta_path = os.path.join(angle_dir, "metadata.json")
        if not os.path.exists(meta_path):
            continue  # Skip directories that don't have the expected file
        with open(meta_path, "r") as f:
            angle_meta = json.load(f)
            angle = angle_meta.get("angle", None)
            predicted_angles.append(angle)

    # Load simulated temperature fields
    simulated_temps = []
    simulated_angles = []
    simulated_masks = []

    # Find all directories in the simulated directory
    sim_angle_dirs = [d for d in os.listdir(simulated_dir) if os.path.isdir(os.path.join(simulated_dir, d))]

    for d in sim_angle_dirs:
        angle_dir = os.path.join(simulated_dir, d)
        
        # Load simulated temperature
        sim_temp_path = os.path.join(angle_dir, "max_T_tp.npy")
        if not os.path.exists(sim_temp_path):
            continue  # Skip directories that don't have the expected file
        simulated_temp = np.load(sim_temp_path)
        simulated_temps.append(simulated_temp)

        # Load mask for simulated data
        mask_path = os.path.join(angle_dir, "inside_outside_array.npy")
        if not os.path.exists(mask_path):
            continue  # Skip directories that don't have the expected file
        mask = np.load(mask_path)[:,2]
        mask = mask.astype(bool)
        mask = mask.reshape(int(np.sqrt(len(mask))), int(np.sqrt(len(mask))))
        simulated_masks.append(mask)

        # Load metadata for simulated data - use "metadata_00.json"
        meta_path = os.path.join(angle_dir, "metadata_00.json")
        if not os.path.exists(meta_path):
            continue  # Skip directories that don't have the expected file
        with open(meta_path, "r") as f:
            angle_meta = json.load(f)
            toolpath_stats = angle_meta.get("toolpath_stats", {})
            raster_angle = toolpath_stats.get("raster_angle_degrees", None)
            simulated_angles.append(raster_angle)

    # Sort predicted results by angle
    if predicted_angles:
        pred_sorted_indices = np.argsort(predicted_angles)
        predicted_temps = [predicted_temps[i] for i in pred_sorted_indices]
        predicted_angles = [predicted_angles[i] for i in pred_sorted_indices]
        predicted_masks = [predicted_masks[i] for i in pred_sorted_indices]

    # Sort simulated results by angle
    if simulated_angles:
        sim_sorted_indices = np.argsort(simulated_angles)
        simulated_temps = [simulated_temps[i] for i in sim_sorted_indices]
        simulated_angles = [simulated_angles[i] for i in sim_sorted_indices]
        simulated_masks = [simulated_masks[i] for i in sim_sorted_indices]

    results = {
        "predicted_temps": predicted_temps,
        "simulated_temps": simulated_temps,
        "predicted_angles": predicted_angles,
        "simulated_angles": simulated_angles,
        "predicted_masks": predicted_masks,
        "simulated_masks": simulated_masks
    }
    return results



# %%
# Example usage:
level = "HIGH"
predicted_dir = f"/mnt/c/Users/jamba/sim_data/OPTIMIZATION_{level}_PA_2/"
simulated_dir = f"/mnt/c/Users/jamba/sim_data/DATABASE_PA_ANGLE_STUDY/{level}_PA/"
results = load_optimization_results(predicted_dir, simulated_dir)

# Adding first element of simulated temps to the end of the simulated results list
results["simulated_temps"].append(results["simulated_temps"][0])
results["simulated_angles"].append(360)
results["simulated_masks"].append(results["simulated_masks"][0])

print("Predicted angles:", results["predicted_angles"])
print("Simulated angles:", results["simulated_angles"])
print("Predicted temperature shape for first angle:", results["predicted_temps"][0].shape if results["predicted_temps"] else None)
print("Simulated temperature shape for first angle:", results["simulated_temps"][0].shape if results["simulated_temps"] else None)

# plot objective value vs angle for predicted and simulated temperatures
import matplotlib.pyplot as plt
def objective(temperature, mask):
    # Apply mask to temperature field before calculating objective
    masked_temp = np.where(mask, temperature, np.nan)
    return np.nanmean(masked_temp)

def objective_std(temperature, mask):
    masked_temp = np.where(mask, temperature, np.nan)
    return np.nanstd(masked_temp)

# plot objective value vs angle for predicted and simulated temperatures
plt.figure(figsize=(12, 6))

# Plot predicted data
if results["predicted_temps"]:
    pred_objectives = [objective(temp, mask) for temp, mask in zip(results["predicted_temps"], results["predicted_masks"])]
    pred_stds = [objective_std(temp, mask) for temp, mask in zip(results["predicted_temps"], results["predicted_masks"])]
    pred_angles = results["predicted_angles"]
    plt.plot(pred_angles, pred_objectives, 'o-', label="Predicted", color='blue')
    pred_objectives = np.array(pred_objectives)
    pred_stds = np.array(pred_stds)
    plt.fill_between(pred_angles, pred_objectives - pred_stds, pred_objectives + pred_stds, color='blue', alpha=0.2)

# Plot simulated data
if results["simulated_temps"]:
    sim_objectives = [objective(temp, mask) for temp, mask in zip(results["simulated_temps"], results["simulated_masks"])]
    sim_stds = [objective_std(temp, mask) for temp, mask in zip(results["simulated_temps"], results["simulated_masks"])]
    sim_angles = results["simulated_angles"]
    plt.plot(sim_angles, sim_objectives, 's-', label="Simulated", color='red')
    sim_objectives = np.array(sim_objectives)
    sim_stds = np.array(sim_stds)
    plt.fill_between(sim_angles, sim_objectives - sim_stds, sim_objectives + sim_stds, color='red', alpha=0.2)

plt.xlabel('Angle (degrees)')
plt.ylabel('Mean Temperature')
plt.title('Objective Value vs Angle')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.show()

# plot temperature field for a selected angle for both predicted and simulated temperatures
# You can change this to any angle you want to visualize
selected_angle = 72.0  # Example: visualize 45 degrees

# Find the closest angle in each dataset to the selected_angle
import numpy as np

# For predicted data
if results["predicted_temps"] and results["predicted_angles"]:
    pred_angles_array = np.array(results["predicted_angles"])
    pred_selected_idx = int(np.argmin(np.abs(pred_angles_array - selected_angle)))
    pred_temp = results["predicted_temps"][pred_selected_idx]
    pred_mask = results["predicted_masks"][pred_selected_idx]
    pred_angle = results["predicted_angles"][pred_selected_idx]
    pred_temp_masked = np.where(pred_mask, pred_temp, np.nan)
else:
    pred_temp_masked = None
    pred_angle = None

# For simulated data
if results["simulated_temps"] and results["simulated_angles"]:
    sim_angles_array = np.array(results["simulated_angles"])
    sim_selected_idx = int(np.argmin(np.abs(sim_angles_array - selected_angle)))
    sim_temp = results["simulated_temps"][sim_selected_idx]
    sim_mask = results["simulated_masks"][sim_selected_idx]
    sim_angle = results["simulated_angles"][sim_selected_idx]
    sim_temp_masked = np.where(sim_mask, sim_temp, np.nan)
else:
    sim_temp_masked = None
    sim_angle = None

# Determine common colorbar limits if both datasets exist
if pred_temp_masked is not None and sim_temp_masked is not None:
    vmin = min(np.nanpercentile(pred_temp_masked, 10), np.nanpercentile(sim_temp_masked, 10))
    vmax = max(np.nanpercentile(pred_temp_masked, 90), np.nanpercentile(sim_temp_masked, 90))
elif pred_temp_masked is not None:
    vmin = np.nanpercentile(pred_temp_masked, 10)
    vmax = np.nanpercentile(pred_temp_masked, 90)
elif sim_temp_masked is not None:
    vmin = np.nanpercentile(sim_temp_masked, 10)
    vmax = np.nanpercentile(sim_temp_masked, 90)
else:
    raise ValueError("No temperature data available for plotting")

plt.figure(figsize=(12, 6))

import matplotlib as mpl

# Create a copy of the 'hot' colormap and set NaN color to light gray
hot_cmap = mpl.cm.get_cmap('hot').copy()
hot_cmap.set_bad(color='lightgray')

if pred_temp_masked is not None:
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(pred_temp_masked, cmap=hot_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im1, label='Temperature')
    plt.title(f"Predicted Temperature at Angle: {pred_angle:.1f}°")
    plt.axis('off')

if sim_temp_masked is not None:
    subplot_idx = 2 if pred_temp_masked is not None else 1
    plt.subplot(1, 2, subplot_idx)
    im2 = plt.imshow(sim_temp_masked, cmap=hot_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im2, label='Temperature')
    plt.title(f"Simulated Temperature at Angle: {sim_angle:.1f}°")
    plt.axis('off')

plt.tight_layout()
plt.show()
# %%