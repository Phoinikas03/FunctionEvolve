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
    
    # Johnson-Cook inspired: sigma = (A + B*eps + C*|eps|^n + D*eps^2) * (1 + e*T + f*T^2)
    # Plus strain-temperature interaction corrections
    
    eps_safe = np.abs(eps) + 1e-12
    n = np.clip(params[3], 0.05, 3.0)
    power_term = params[2] * np.sign(eps) * eps_safe ** n
    
    # Mechanical response (strain hardening)
    mechanical = params[0] + params[1] * eps + power_term + params[4] * eps * eps
    
    # Thermal softening (multiplicative, quadratic in T)
    thermal = 1.0 + params[5] * T + params[6] * T * T
    
    # Base result: multiplicative coupling
    result = mechanical * thermal
    
    # Additive strain-temperature interactions for residual patterns
    result = result + params[7] * eps * T + params[8] * eps * eps * T
    
    # Additional T offset
    result = result + params[9] * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
