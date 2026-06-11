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
    
    # Extended population dynamics model:
    # Combines logistic, Gompertz, Allee, and time-varying effects
    
    P_safe = np.maximum(P, 1e-10)
    log_P = np.log(P_safe)
    
    # Core growth terms
    # params[0]: constant immigration/base rate
    # params[1]*P: exponential growth component
    # params[2]*P^2: logistic density-dependent term
    # params[3]*P*ln(P): Gompertz-type term
    # params[4]*t*P: time-varying growth rate (r changes with t)
    # params[5]*t*P*ln(P): time-varying Gompertz
    # params[6]*P^3: higher-order density dependence
    # params[7]*t: linear time trend
    # params[8]*exp(-params[9]*t)*P: exponentially decaying growth rate
    
    P2 = P * P
    tP = t * P
    
    result = (params[0]
              + params[1] * P
              + params[2] * P2
              + params[3] * P * log_P
              + params[4] * tP
              + params[5] * t * P * log_P
              + params[6] * P2 * P
              + params[7] * t
              + params[8] * P / (np.abs(params[9]) + P_safe + 1e-10))
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
