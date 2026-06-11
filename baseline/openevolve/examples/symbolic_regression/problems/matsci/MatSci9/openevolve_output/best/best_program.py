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
    
    # Protect against numerical issues with power law
    eps_safe = np.abs(eps) + 1e-12
    
    # Power-law strain hardening: (p0 + p1 * eps^p2)
    # p2 is typically between 0 and 1 for work hardening
    n_exp = np.clip(params[2], 0.01, 5.0)
    strain_term = params[0] + params[1] * np.power(eps_safe, n_exp)
    
    # Temperature softening: exponential decay + linear
    temp_factor = np.exp(-params[3] * T) + params[4] * T + params[5]
    
    # Combined model: strain hardening * temperature factor + offset
    # Plus strain-temperature interaction
    result = strain_term * temp_factor + params[6] * eps * T + params[7]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
