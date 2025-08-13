import time
import sys
import numpy as np
from typing import Tuple, List

# Ensure project root is on path
sys.path.append('/home/jamba/research/thesis-v2/')

from shapely.geometry import Polygon

from geometry.geometry_samples import create_square, generate_grid_points
from toolpath.path_gen import generate_toolpath
from solver.config import get_mesh_size_presets
from toolpath.fields import calculate_signed_distance_field

from scipy.interpolate import RBFInterpolator

import torch

from mLearning.models.cnn_16 import myModel


def timed_preprocess_input_fields(
    polygon: Polygon,
    toolpath: np.ndarray,
    *,
    discretization_length: float,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, int, float]:
    """
    Lightly modified version of solver.preprocess.preprocess_input_fields with timing
    and support for padding the evaluation domain.

    Returns (raw_fields, inside_mask, all_points, grid_size, feature_calc_time_seconds)
    where raw_fields = [sdf_field, time_field, grad_mag].
    """
    t0 = time.perf_counter()

    # Generate evaluation points (grid points within context square)
    all_points = generate_grid_points(polygon, discretization_length)
    grid_points = all_points[:, :2]
    grid_size = int(np.sqrt(len(grid_points)))
    if grid_size < 8:
        raise ValueError(f"Grid size {grid_size} is too small for U-Net. Minimum size is 8x8.")

    # SDF
    sdf_values = calculate_signed_distance_field(grid_points, polygon)
    sdf_field = sdf_values.reshape((grid_size, grid_size))

    # Time field via RBF
    toolpath_points = toolpath[:, 1:3]
    toolpath_times = toolpath[:, 0]
    rbf_interp = RBFInterpolator(toolpath_points, toolpath_times, kernel='linear')
    time_values = rbf_interp(grid_points)
    time_field = time_values.reshape((grid_size, grid_size))

    # Gradient magnitude of time field
    grad_y, grad_x = np.gradient(time_field)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Inside mask
    inside_mask = all_points[:, 2].reshape((grid_size, grid_size)).astype(bool)

    raw_fields = [sdf_field, time_field, grad_mag]

    t1 = time.perf_counter()
    return raw_fields, inside_mask, all_points, grid_size, (t1 - t0)


def format_seconds_ms(s: float) -> str:
    return f"{s * 1e3:.1f} ms"


def main():
    # Discretization length: keep constant using laser region spacing from 'fine' preset
    discretization_length = get_mesh_size_presets('fine')['laser']

    # Sweep square lengths (meters). Keep ascending so last is the largest domain.
    lengths_m = [
        0.5e-3,
        1.0e-3,
        2.0e-3,
        4.0e-3,
        8.0e-3,
    ]

    # Warm-up: run one feature calculation to initialize SciPy/BLAS threadpools, etc.
    warm_shape = create_square(size=1.0e-3, hollowness=0.0)
    warm_toolpath = generate_toolpath(warm_shape, pattern="uniraster", raster_angle=0.0)
    _ = timed_preprocess_input_fields(
        warm_shape, warm_toolpath, discretization_length=discretization_length
    )

    # Model setup (CPU by default for consistent timing)
    torch.set_grad_enabled(False)
    device = torch.device('cpu')
    model = myModel().to(device).eval()

    rows = []
    for length in lengths_m:
        # Shape and toolpath for this length
        shape = create_square(size=length, hollowness=0.0)
        toolpath = generate_toolpath(shape, pattern="uniraster", raster_angle=0.0)

        # Feature calculations (timed)
        raw_fields, inside_mask, all_points, grid_size, fc_time = timed_preprocess_input_fields(
            shape, toolpath, discretization_length=discretization_length
        )

        # Prepare input tensor [1, 3, H, W]
        x = np.stack(raw_fields, axis=0).astype(np.float32)
        x = torch.from_numpy(x[None, ...]).to(device)

        # Inference timing (single forward)
        t0 = time.perf_counter()
        _ = model(x)
        t1 = time.perf_counter()
        inf_time = (t1 - t0)

        rows.append({
            'domain': f"{grid_size}x{grid_size}",
            'length_m': length,
            'fc_time': fc_time,
            'inf_time': inf_time,
        })

    # Print Markdown table
    print("| Domain Size | Length (mm) | Feature Calculations | Inference Time | Total Time |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        length_mm = r['length_m'] * 1e3
        total_time = r['fc_time'] + r['inf_time']
        print(
            f"| {r['domain']} | {length_mm:.3f} | {format_seconds_ms(r['fc_time'])} | {format_seconds_ms(r['inf_time'])} | {format_seconds_ms(total_time)} |"
        )


if __name__ == "__main__":
    main()

