"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dP_dt (Population growth rate)
Input variables (columns of x): t (Time), P (Population at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: t (Time)
#   x[:, 1]: P (Population at time t)

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
                        (t, P).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    t = x[:, 0]
    P = x[:, 1]
    
    # Population growth model combining logistic, Gompertz, and time effects
    # Safeguard for log
    P_safe = np.maximum(np.abs(P), 1e-10)
    
    # Core logistic terms
    result = params[0] + params[1] * P + params[2] * P * P
    
    # Gompertz-like term
    result = result + params[3] * P * np.log(P_safe)
    
    # Time-varying growth rate
    result = result + params[4] * t * P + params[5] * t
    
    # Exponential decay modulation of growth
    result = result + params[6] * P * np.exp(-params[7] * t * t)
    
    # Saturation term
    result = result + params[8] * P / (P_safe + np.abs(params[9]) + 1e-6)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
