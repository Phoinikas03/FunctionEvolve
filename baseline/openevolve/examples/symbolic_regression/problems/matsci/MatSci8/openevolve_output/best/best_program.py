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
    epsilon = x[:, 0]
    T = x[:, 1]
    
    eps_safe = np.abs(epsilon) + 1e-12
    
    # Power-law exponent via sigmoid: maps params[2] to (0.05, 2.0)
    n_exp = 0.05 + 1.95 / (1.0 + np.exp(-params[2]))
    
    # Power-law hardening
    eps_pow = eps_safe ** n_exp
    power_hard = params[1] * eps_pow
    
    # Voce saturation with temperature-dependent rate
    # Higher T -> faster saturation (dynamic recovery)
    voce_rate = params[3] + params[7] * T
    exp_arg = np.clip(-voce_rate * eps_safe, -50, 50)
    voce = params[4] * (1.0 - np.exp(exp_arg))
    
    # Combined strain hardening with yield stress
    strain_part = params[0] + power_hard + voce
    
    # Temperature softening: exponential decay + linear
    temp_exp_arg = np.clip(-params[5] * T, -50, 50)
    temp_part = np.exp(temp_exp_arg) + params[6] * T
    
    # Multiplicative coupling
    result = strain_part * temp_part
    
    # Power-law strain * T cross-coupling
    result = result + params[8] * eps_pow * T
    
    # Offset
    result = result + params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
