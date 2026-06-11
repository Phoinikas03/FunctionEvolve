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
    
    eps_abs = np.abs(eps) + 1e-10
    
    # Power-law strain hardening
    n_exp = np.clip(params[2], 0.05, 3.0)
    power_hard = params[1] * eps_abs ** n_exp
    
    # Voce-type saturating hardening
    voce = params[3] * (1.0 - np.exp(-np.clip(params[4] * eps_abs, 0, 50)))
    
    # Combined strain hardening
    strain_part = params[0] + power_hard + voce
    
    # Temperature softening: constant + exponential decay + linear
    temp_exp = np.exp(-np.clip(params[7] * T, -50, 50))
    temp_part = params[5] + params[6] * temp_exp + params[8] * T
    
    # Multiplicative coupling
    result = strain_part * temp_part
    
    # Offset
    result = result + params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
