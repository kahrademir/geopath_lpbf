"""
3D Thermal Analysis Package for Laser Powder Bed Fusion

This package contains modules for simulating the thermal behavior during
additive manufacturing processes.

Modules:
- config: Configuration parameters and constants
- utils: Utility functions for I/O and MPI operations  
- material_models: Material property models (e.g., temperature-dependent specific heat)
- heat_source: Laser heat source and toolpath generation
- mesh_utils: Mesh generation and loading utilities
- thermal_solver: Main thermal problem setup and solver
"""

from .config import *
from .utils import mpi_print
from .thermal_solver import simulate

__version__ = "1.0.0"
__author__ = "K.G.D." 