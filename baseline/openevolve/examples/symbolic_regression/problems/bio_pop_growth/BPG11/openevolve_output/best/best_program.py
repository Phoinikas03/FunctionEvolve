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
    
    # Generalized population growth model combining logistic, Gompertz, and Allee effects
    P_safe = np.clip(P, 1e-10, 1e10)
    log_P = np.log(P_safe)
    sqrt_P = np.sqrt(P_safe)
    
    # Exponential decay rate (ensure non-negative decay)
    decay_rate = np.abs(params[8])
    exp_term = np.exp(-decay_rate * np.clip(t, 0, 100))
    
    result = (params[0]                          # intercept (immigration/constant)
              + params[1] * P_safe               # linear growth (r*P)
              + params[2] * P_safe**2             # logistic saturation (-r/K * P^2)
              + params[3] * P_safe * log_P        # Gompertz term: P*ln(P)
              + params[4] * t                     # time trend
              + params[5] * t * P_safe            # time-varying growth rate
              + params[6] * sqrt_P                # sqrt term (Allee-like, sub-linear growth at low P)
              + params[7] * exp_term * P_safe     # transient growth dynamics
              + params[9] * t * log_P             # time-log interaction
              )
    
    return np.clip(result, -1e15, 1e15)
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
