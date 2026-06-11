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
    
    # Power-law strain hardening with temperature softening (Johnson-Cook inspired)
    # sigma = (A + B * eps^n) * (C + D*T) + linear corrections
    
    # Safe power law: |eps|^p2 * sign(eps) to handle negative/zero strain
    abs_eps = np.abs(eps) + 1e-12
    n_exp = np.clip(params[2], 0.01, 5.0)  # hardening exponent, bounded
    eps_power = np.power(abs_eps, n_exp) * np.sign(eps)
    
    # Hardening term: (p0 + p1 * eps^n)
    hardening = params[0] + params[1] * eps_power
    
    # Temperature softening: (p3 + p4 * T)
    temp_factor = params[3] + params[4] * T
    
    # Combined constitutive model + linear corrections
    result = (hardening * temp_factor 
              + params[5] * eps 
              + params[6] * T 
              + params[7] * eps * T 
              + params[8] * eps**2
              + params[9])
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
