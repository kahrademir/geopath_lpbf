import matplotlib.pyplot as plt
import numpy as np
from typing import List

def plot_toolpath(toolpath: np.ndarray, 
                     fields: List[np.ndarray] = None, 
                     field_names: List[str] = None,
                     point_size: float = 40,
                     colormap: str = 'hot',
                     show: bool = True):
    """
    Plots a toolpath. If fields are provided, they are plotted as scatter plots side-by-side.

    Args:
        toolpath: numpy array of shape (n_points, 3) containing [time, x, y]
        fields: List of numpy arrays of shape (n_points, 1 or 2) containing either scalar or vector fields
        field_names: List of strings containing the names of the fields
        point_size: Size of scatter plot points
        colormap: Colormap to use for field visualization
        show: Whether to immediately display the plot (default: True)
        
    Returns:
        fig: matplotlib figure object
        axes: matplotlib axes object(s) - single axis if no fields, list of axes if fields provided
    """
    if fields is not None:
        fig, axes = plt.subplots(1, len(fields), figsize=(8*len(fields), 8))
        if len(fields) == 1:
            axes = [axes]
        
        for i, (field, field_name) in enumerate(zip(fields, field_names)):

            ax = axes[i]
            ax.set_facecolor('lightgray')  # Set background to gray
            # Plot toolpath line first
            ax.plot(toolpath[:, 1], toolpath[:, 2], 'b-', linewidth=1)
            
            # Add start and stop markers
            start_idx = 0
            stop_idx = -1
            ax.scatter(toolpath[start_idx, 1], toolpath[start_idx, 2], marker='*', color='green', s=120, label='Start')
            ax.scatter(toolpath[stop_idx, 1], toolpath[stop_idx, 2], marker='X', color='red', s=120, label='Stop')
            
            # # Calculate 95th percentile limits to exclude outliers
            # vmin = np.percentile(field, 5)
            # vmax = np.percentile(field, 95)
            vmin = np.percentile(field, 2)
            vmax = np.percentile(field, 98)
            
            # Plot field values as scatter plot on top
            scatter = ax.scatter(toolpath[:, 1], toolpath[:, 2], c=field, cmap=colormap, s=point_size, vmin=vmin, vmax=vmax)
            plt.colorbar(scatter, ax=ax, label=field_name)
            
            ax.set_title(f'{field_name} along Toolpath')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.grid(True)
            ax.legend()
            ax.axis('equal')
            # plt.show()

    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(toolpath[:, 1], toolpath[:, 2], 'b-', linewidth=1)
        
        start_idx = 0
        stop_idx = -1
        ax.scatter(toolpath[start_idx, 1], toolpath[start_idx, 2], marker='*', color='green', s=120, label='Start')
        ax.scatter(toolpath[stop_idx, 1], toolpath[stop_idx, 2], marker='X', color='red', s=120, label='Stop')
        ax.set_title('Laser Toolpath')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True)
        ax.axis('equal')
        ax.legend()
        axes = ax  # Return single axis, not list
    
    plt.tight_layout()
    
    if show:
        plt.show()
    
    return fig, axes

def test():
    # # from geometry_samples import generate_shapes
    import sys
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.insert(0, parent_dir)

    toolpath_dir = "/mnt/c/Users/jamba/sim_data/DATABASE_PATTERNS/DATA_003"

    toolpath_file = os.path.join(toolpath_dir, "toolpath.npy")
    if not os.path.exists(toolpath_file):
        raise FileNotFoundError(f"Toolpath file not found: {toolpath_file}")
    toolpath = np.load(toolpath_file)
    plot_toolpath(toolpath, colormap='bwr', point_size=50)

    plt.show()






if __name__ == "__main__":
    test()