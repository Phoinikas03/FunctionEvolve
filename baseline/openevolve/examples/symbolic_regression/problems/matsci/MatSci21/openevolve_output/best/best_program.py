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
    
    # Power-law strain hardening with temperature-dependent exponent
    n_exp = np.clip(params[2] + params[6] * T, 0.05, 4.0)
    power_term = params[1] * np.power(abs_eps, n_exp)
    
    # Voce-type saturation: sigma_sat * (1 - exp(-rate * eps))
    # More physical than tanh for work hardening saturation
    voce_arg = np.clip(-params[9] * abs_eps, -50.0, 0.0)
    voce_term = params[8] * (1.0 - np.exp(voce_arg))
    
    # Combined hardening (without yield stress - that goes in linear part)
    hardening = power_term + voce_term
    
    # Temperature softening: exponential decay (Johnson-Cook inspired)
    exp_arg = np.clip(-params[4] * T, -50.0, 50.0)
    temp_factor = params[3] * np.exp(exp_arg) + params[5]
    
    # Stress = linear_term + hardening * temp_softening
    # Linear term captures elastic regime; eps (not |eps|) preserves sign
    result = params[0] * eps * temp_factor + hardening * np.sign(eps) * temp_factor
    
    # Constant offset
    result = result + params[7]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
