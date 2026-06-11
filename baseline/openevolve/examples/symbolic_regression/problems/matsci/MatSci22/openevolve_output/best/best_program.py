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
    
    # Power-law strain hardening with free exponent
    n_exp = params[3]
    power_term = np.sign(eps) * eps_abs ** n_exp
    
    # Strain hardening: constant + linear + power-law
    strain_effect = params[0] + params[1] * eps + params[2] * power_term
    
    # Thermal softening: linear * exponential (physically motivated)
    temp_effect = (1.0 + params[4] * T) * np.exp(params[5] * T)
    
    # Main model: strain hardening modulated by temperature
    result = strain_effect * temp_effect
    
    # Additive corrections
    result = result + params[6] * eps * T
    result = result + params[7]
    result = result + params[8] * eps * eps * T
    result = result + params[9] * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
