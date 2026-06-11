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
    
    eps_safe = np.abs(eps) + 1e-12
    
    # Smooth exponent via softplus: always > 0.05, smooth for BFGS
    n = np.log1p(np.exp(np.clip(params[4], -10, 10))) + 0.05
    
    # Power-law hardening
    eps_power = np.power(eps_safe, n) * np.sign(eps)
    
    # Voce-type saturating hardening with temperature-dependent rate
    # Higher T -> different saturation behavior (physically motivated)
    rate = np.clip(params[7] + params[8] * T, -50.0, 50.0)
    voce = 1.0 - np.exp(-rate * eps_safe)
    
    # Base stress-strain: constant + linear + power-law + Voce
    base = params[0] + params[1] * eps + params[2] * eps_power + params[3] * voce
    
    # Exponential thermal softening + linear correction
    exp_arg = np.clip(-params[5] * T, -50.0, 50.0)
    temp_factor = np.exp(exp_arg) + params[6] * T
    
    # Combined: multiplicative thermal + offset
    result = base * temp_factor + params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
