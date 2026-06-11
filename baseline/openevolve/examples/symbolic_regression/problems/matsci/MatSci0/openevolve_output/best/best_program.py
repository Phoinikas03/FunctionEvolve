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
    eps = x[:, 0]
    T = x[:, 1]
    
    eps_abs = np.abs(eps) + 1e-12
    sgn = np.sign(eps + 1e-30)
    
    # Power-law hardening exponent (positive, temperature-dependent)
    n = params[2] ** 2 + 0.01
    
    # Voce saturation rate (positive)
    k = params[5] ** 2 + 0.01
    
    # Strain hardening components (all odd in epsilon)
    linear_term = params[0] * eps
    power_term = params[1] * eps_abs ** n * sgn
    voce_term = params[6] * np.tanh(k * eps)
    
    strain_term = linear_term + power_term + voce_term
    
    # Thermal softening: (1 + a*T + b*T^2) allows flexible T dependence
    thermal_mult = 1.0 + params[3] * T + params[4] * T * T * 1e-4
    
    # Multiplicative coupling
    result = strain_term * thermal_mult
    
    # Temperature-dependent power-law correction (odd in eps)
    # Captures change in hardening character with temperature
    result = result + params[7] * eps_abs ** 0.5 * sgn * T
    
    # Strain-temperature interaction (odd in epsilon)
    result = result + params[8] * eps * T * T * 1e-3
    
    # Additive temperature term
    result = result + params[9] * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
