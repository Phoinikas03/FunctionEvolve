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
    
    # Constitutive model with T-dependent hardening mechanisms
    # Physically: yield stress, hardening rate, and saturation all depend on T
    
    abs_eps = np.abs(eps) + 1e-12
    
    # T-dependent yield stress
    sigma_y = params[0] + params[1] * T
    
    # Voce saturation: sigma_sat*(1 - exp(-theta*eps)) with T-dependent parameters
    # Saturation stress depends on T
    sigma_sat = params[2] + params[3] * T
    # Saturation rate depends on T
    theta = params[4] * (1.0 + params[5] * T)
    voce_arg = np.clip(-np.abs(theta) * abs_eps, -50.0, 0.0)
    voce_term = sigma_sat * np.sign(eps) * (1.0 - np.exp(voce_arg))
    
    # sqrt hardening for initial rapid rise (T-dependent coefficient)
    sqrt_coeff = params[6] + params[7] * T
    sqrt_term = sqrt_coeff * np.sign(eps) * np.sqrt(abs_eps)
    
    # Combined stress
    result = sigma_y + voce_term + sqrt_term + params[8] * eps * T + params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
