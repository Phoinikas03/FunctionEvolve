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
    
    # Strain hardening component (polynomial + power law inspired)
    # params[0]: intercept (yield stress)
    # params[1]: linear strain coefficient
    # params[2]: quadratic strain coefficient
    # params[3]: power-law exponent for strain
    
    abs_eps = np.abs(eps) + 1e-12
    
    # Temperature-dependent power-law exponent
    n = np.clip(params[3] + params[9] * T, 0.05, 3.0)
    eps_power = np.sign(eps) * np.power(abs_eps, n)
    
    # Power-law hardening + Voce saturation (two hardening mechanisms)
    rate = np.clip(params[7], 0.01, 100.0)
    voce = params[8] * (1.0 - np.exp(-rate * abs_eps))
    base_stress = params[0] + params[1] * eps_power + voce
    
    # Multiplicative thermal softening factor (polynomial in T)
    thermal_factor = 1.0 + params[4] * T + params[5] * T * T
    
    # Core: multiplicative coupling
    result = base_stress * thermal_factor
    
    # Additive strain-temperature interaction + linear strain term
    result += params[2] * eps * T + params[6] * eps
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
