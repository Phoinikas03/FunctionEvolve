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
    
    # Protect against numerical issues
    P_safe = np.maximum(np.abs(P), 1e-10)
    log_P = np.log(P_safe)
    inv_P = 1.0 / P_safe
    P2 = P * P
    log_P2 = log_P * log_P
    
    # Multiplicative population dynamics: P * (growth - density_dep)
    # with additive corrections for Allee and time effects
    # Core: Gompertz-logistic hybrid with time-varying carrying capacity
    # dP/dt = P * [a1 + a2*ln(P) + a4*t + a5*t*ln(P)] + a0 + a6*P² + a7*P³ + a8/P + a9*P*ln²(P)
    growth_rate = params[1] + params[2] * log_P + params[3] * t + params[5] * t * log_P
    result = (params[0]
              + P * growth_rate
              + params[4] * P2
              + params[6] * P2 * P
              + params[7] * inv_P
              + params[8] * P * log_P2
              + params[9] * t * P2)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
