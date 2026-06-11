"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: sigma (Stress)
Input variables (columns of x): epsilon (Strain), T (Temperature)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: epsilon (Strain)
#   x[:, 1]: T (Temperature)

# Parameters will be optimized by BFGS outside this function.
# Number of parameters expected by this model: 10.
# Example initialization: params = np.random.rand(10)

# EVOLVE-BLOCK-START

def func(x, params):
    """
    Calculates the model output using a linear combination of input variables
    or a constant value if no input variables. Operates on a matrix of samples.

    Args:
        x (np.ndarray): A 2D numpy array of input variable values, shape (n_samples, n_features).
                        n_features is 2.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        (epsilon, T).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    eps = x[:, 0]  # strain
    T = x[:, 1]    # temperature
    
    # Physically-motivated stress model:
    # Two Voce-like saturation terms (dual mechanism) with thermal softening
    
    abs_eps = np.abs(eps) + 1e-12
    
    # Power-law hardening: A + B * |eps|^n
    n_exp = np.clip(params[2], 0.05, 3.0)
    hardening = params[0] + params[1] * np.power(abs_eps, n_exp)
    
    # Voce saturation with T-dependent rate: higher T -> faster saturation
    # Rate parameter: |p7| + p8*T  (temperature accelerates dynamic recovery)
    inv_scale = np.abs(params[7]) + 1e-8 + params[8] * T
    inv_scale_safe = np.clip(np.abs(inv_scale) + 1e-8, 1e-8, 200.0)
    voce = params[6] * (1.0 - np.exp(-np.clip(inv_scale_safe * abs_eps, 0, 50)))
    
    # Thermal softening: polynomial (multiplicative)
    thermal_factor = 1.0 + params[3] * T + params[4] * T * T
    
    # Combined with eps*T interaction for cross-coupling
    result = (hardening + voce) * thermal_factor + params[5] + params[9] * eps * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
