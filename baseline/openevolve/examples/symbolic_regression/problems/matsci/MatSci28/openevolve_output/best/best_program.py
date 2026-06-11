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
    
    eps_abs = np.abs(eps) + 1e-12
    s = np.sign(eps)
    
    # Temperature-dependent power-law exponent
    n = np.clip(params[2] + params[3] * T, 0.01, 5.0)
    
    # Power-law strain hardening with temperature-dependent exponent
    strain_hardening = params[0] + params[1] * np.power(eps_abs, n) * s
    
    # Temperature softening (quadratic polynomial)
    temp_factor = 1.0 + params[4] * T + params[5] * T * T
    
    # Multiplicative coupling
    result = strain_hardening * temp_factor
    
    # Voce-type saturation hardening (odd in strain, temperature-dependent rate)
    p8 = np.clip(params[8], -50.0, 50.0)
    voce = params[7] * (1.0 - np.exp(-p8 * eps_abs)) * s
    
    # Temperature-dependent Voce amplitude scaling
    result = result + voce * (1.0 + params[9] * T)
    
    # Strain-temperature interaction + offset
    result = result + params[6] * eps * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
