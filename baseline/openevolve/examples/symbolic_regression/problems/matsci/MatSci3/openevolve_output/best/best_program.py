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
    
    # Stress model with sqrt and cbrt for flexible hardening shape
    # sqrt (n=0.5) and cbrt (n=0.33) span typical metal hardening exponents
    
    eps_safe = np.abs(eps) + 1e-12
    sqrt_eps = np.sqrt(eps_safe)
    cbrt_eps = np.cbrt(eps)
    eps2 = eps * eps
    T2 = T * T
    
    # Strain hardening: constant + linear + quadratic + sqrt + cbrt
    strain_part = params[0] + params[1] * eps + params[2] * eps2 \
                + params[3] * sqrt_eps + params[4] * cbrt_eps
    
    # Multiplicative temperature softening (quadratic in T)
    temp_factor = 1.0 + params[5] * T + params[6] * T2
    
    # Cross terms: fractional-power strain-temperature coupling
    cross_terms = params[7] * sqrt_eps * T + params[8] * cbrt_eps * T
    
    result = strain_part * temp_factor + cross_terms + params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
