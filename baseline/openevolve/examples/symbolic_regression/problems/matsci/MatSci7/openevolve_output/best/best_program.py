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
    
    # Enhanced Johnson-Cook with T-dependent hardening exponent
    # sigma = (A + B*|eps|^n(T) + C*tanh(D*eps)) * (1 + E*T + F*T^2) + interactions
    
    eps_safe = np.abs(eps) + 1e-12
    
    # Temperature-dependent strain hardening exponent
    n_exp = np.clip(params[4] + params[9] * T, 0.05, 4.0)
    
    # Strain hardening: yield + power-law(T-dependent n) + tanh saturation
    strain_term = params[0] + params[1] * (eps_safe ** n_exp) + params[5] * np.tanh(params[6] * eps)
    
    # Temperature softening: polynomial
    temp_term = 1.0 + params[2] * T + params[3] * T**2
    
    # Multiplicative coupling
    result = strain_term * temp_term
    
    # Strain-temperature interaction
    result = result + params[7] * eps * T
    
    # Offset
    result = result + params[8]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
