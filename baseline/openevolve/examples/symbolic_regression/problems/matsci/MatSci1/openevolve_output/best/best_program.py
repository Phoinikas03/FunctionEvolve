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
    
    # Smooth positive exponent via softplus (better BFGS gradients)
    p2_safe = np.clip(params[2], -10.0, 10.0)
    n_exp = np.log1p(np.exp(p2_safe)) + 0.05
    n_exp = min(n_exp, 3.0)
    eps_power = np.sign(eps) * np.power(abs_eps, n_exp)
    
    # Strain hardening: yield + power law
    strain_term = params[0] + params[1] * eps_power
    
    # Quadratic temperature softening (multiplicative)
    temp_mod = 1.0 + params[3] * T + params[4] * T * T
    
    # Multiplicative coupling (Johnson-Cook style)
    result = params[5] * strain_term * temp_mod
    
    # Voce saturating hardening with temperature dependence
    p8_safe = np.clip(params[8], -10.0, 10.0)
    rate = np.log1p(np.exp(p8_safe)) + 0.01
    voce_arg = np.clip(-rate * abs_eps, -50.0, 0.0)
    voce = 1.0 - np.exp(voce_arg)
    result = result + params[6] * voce * (1.0 + params[9] * T)
    
    # Tanh term: captures elastic-to-plastic transition smoothly
    # tanh(a*eps) ~ a*eps for small eps (linear elastic), saturates for large eps
    result = result + params[7] * np.tanh(rate * eps)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
