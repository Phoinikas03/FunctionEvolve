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
    
    # Population growth model combining:
    # - Logistic core: r*P*(1 - P/K) via params[1]*P + params[2]*P^2
    # - Saturating/rational term: params[5]*P / (params[6]^2 + P^2) for Allee-like effects
    # - Time-varying growth: params[7]*P*exp(-params[8]*t)
    # - Time-dependent carrying capacity: params[4]*t*P + params[9]*t*P^2
    
    P_safe = np.abs(P) + 1e-10
    log_P = np.log(P_safe)
    log_P2 = log_P * log_P
    rate = np.clip(params[8], -5, 5)
    exp_term = np.exp(-rate * t)
    K_sq = params[4] * params[4] + 1e-10
    sat_denom = 1.0 + K_sq * P_safe
    
    # Hybrid model: Beverton-Holt saturation + Gompertz hierarchy + time effects
    # Core structure from best model (3.2804) with stabilized denominator
    # Added: t*P term for direct time-population interaction
    result = (params[0]
              + params[1] * P
              + params[2] * P * P
              + params[3] * P / sat_denom       # Beverton-Holt saturation
              + params[5] * P * log_P           # Gompertz
              + params[6] * P * log_P2          # Higher-order Gompertz
              + params[7] * P * exp_term        # Time-decaying growth
              + params[9] * t * P * log_P)      # Time-varying Gompertz
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
