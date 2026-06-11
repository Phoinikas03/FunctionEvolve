"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dA_dt (Rate of change of concentration in chemistry reaction kinetics)
Input variables (columns of x): t (Time), A (Concentration at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: t (Time)
#   x[:, 1]: A (Concentration at time t)

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
                        (t, A).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    t = x[:, 0]
    A = x[:, 1]
    A_safe = np.clip(A, 1e-30, None)
    
    log_A = np.log(A_safe)
    
    # Polynomial in A: constant + first-order + second-order
    poly = params[0] + params[1] * A + params[2] * A**2
    
    # Time terms with quadratic time dependence
    time_terms = params[3] * t + params[4] * t * A
    
    # Power-law kinetics: k * A^n (generalized nth-order)
    n = np.clip(params[6], 0.01, 5.0)
    power_term = params[5] * A_safe**n
    
    # Log term captures different curvature than polynomial/power
    log_term = params[7] * log_A
    
    # A*log(A) term: captures autocatalytic kinetics
    Alog_term = params[8] * A * log_A
    
    # Exponential decay in A: captures sharp transitions at low concentrations
    exp_term = params[9] * np.exp(-np.clip(A * np.abs(params[6]), 0, 30))
    
    result = poly + time_terms + power_term + log_term + Alog_term + exp_term
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
