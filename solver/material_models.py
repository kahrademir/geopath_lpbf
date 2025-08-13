#!/usr/bin/env python3
"""
Material models for 3D thermal analysis
"""

import numpy as np
from .config import *

def effective_specific_heat(T):
    """
    Calculate effective specific heat capacity including latent heat of fusion
    using the apparent heat capacity method from literature.
    
    Based on equations:
    C_p(θ) = C_n(θ) + C_ap(θ)
    C_n(θ) = C_s + (C_l - C_s)/2 * (1 + tanh(8 * (θ - θ_m)/(θ_2 - θ_1)))
    C_ap(θ) = 2L/(θ_2 - θ_1) * cos²(π * (θ - θ_m)/(θ_2 - θ_1))
    
    Args:
        T: Temperature (K) - can be scalar or array
        
    Returns:
        cp_eff: Effective specific heat capacity (J/kg·K)
    """
    # Material properties
    C_s = SPECIFIC_HEAT  # Solid specific heat capacity
    C_l = SPECIFIC_HEAT  # Liquid specific heat capacity (often similar to solid)
    
    # Temperature bounds for latent heat effects
    theta_m = MELTING_TEMP  # Melting temperature
    theta_1 = theta_m - MELTING_RANGE / 2  # Lower bound
    theta_2 = theta_m + MELTING_RANGE / 2  # Upper bound
    L = LATENT_HEAT  # Latent heat of fusion
    
    # Equation (7): Temperature-dependent specific heat
    tanh_arg = 8 * (T - theta_m) / (theta_2 - theta_1)
    C_n = C_s + (C_l - C_s) / 2 * (1 + np.tanh(tanh_arg))
    
    # Equation (8): Apparent heat capacity for latent heat
    cos_arg = np.pi * (T - theta_m) / (theta_2 - theta_1)
    C_ap = (2 * L) / (theta_2 - theta_1) * np.cos(cos_arg)**2
    
    # Only apply C_ap within the melting range
    mask = (T >= theta_1) & (T <= theta_2)
    C_ap = np.where(mask, C_ap, 0.0)
    
    # Total effective specific heat capacity
    cp_eff = C_n + C_ap
    
    return cp_eff 