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
    
    # Logistic-type population growth model with extensions
    # Core: dP/dt = r*P*(1 - P/K) = r*P - (r/K)*P^2
    # Extended with constant, time effects, and higher-order terms
    
    # Protect against overflow
    P_safe = np.clip(P, -1e10, 1e10)
    
    # Safe exponential transient
    exp_arg = np.clip(-params[8]**2 * t, -100, 0)
    exp_term = np.exp(exp_arg)
    
    # Precompute powers
    P2 = P_safe**2
    
    # Saturating cubic: P^3/(1 + a*P^2) for graceful large-P behavior
    denom_cubic = 1.0 + params[6]**2 * P2
    
    # Saturating growth: P/(1 + b*P) - Michaelis-Menten / Beverton-Holt type
    abs_P = np.abs(P_safe) + 1e-10
    denom_mm = 1.0 + params[9]**2 * abs_P
    mm_term = P_safe / denom_mm
    
    result = (params[0]                              # constant offset
              + params[1] * P_safe                   # linear growth r*P
              + params[2] * P2                       # density dependence -r/K * P^2
              + params[3] * t                        # time trend
              + params[4] * t * P_safe               # time-varying growth rate
              + params[5] * P_safe * P2 / denom_cubic  # saturating cubic
              + params[7] * exp_term * P_safe        # transient growth component
              + params[9] * P_safe * exp_term * t    # time-decaying growth interaction
              )
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
