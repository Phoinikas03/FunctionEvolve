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
    
    abs_eps = np.abs(eps) + 1e-12
    sgn = np.sign(eps)
    
    # Power-law hardening: |eps|^n * sign(eps)
    n_exp = np.clip(params[3], 0.01, 5.0)
    eps_power = sgn * (abs_eps ** n_exp)
    
    # Tanh saturation term (smooth Voce-like) with temperature-dependent rate
    base_rate = np.clip(params[8], 0.001, 200.0)
    tanh_term = np.tanh(base_rate * eps)
    
    # Strain contribution: yield + linear + power law + saturation
    strain_part = params[0] + params[1] * eps + params[2] * eps_power + params[9] * tanh_term
    
    # Temperature softening: multiplicative (1 + p4*T + p5*T^2)
    temp_factor = 1.0 + params[4] * T + params[5] * T * T
    
    # Combined model
    result = strain_part * temp_factor + params[6] * T + params[7]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
