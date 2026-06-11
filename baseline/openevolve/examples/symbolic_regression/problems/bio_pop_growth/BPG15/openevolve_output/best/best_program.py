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
    
    P_safe = np.clip(P, -1e10, 1e10)
    P2 = P_safe * P_safe
    abs_P = np.abs(P_safe) + 1e-10
    log_P = np.log(abs_P)
    
    # Half-saturation constant (positive via squaring)
    s1 = params[7]**2 + 1e-10
    
    # Holling type II: P / (half_sat + |P|)
    holling = P_safe / (s1 + abs_P)
    
    # Ricker-like: P * exp(-s*|P|) for density-dependent growth
    ricker_scale = params[9]**2 + 1e-10
    ricker_arg = np.clip(-ricker_scale * abs_P, -50, 50)
    ricker = P_safe * np.exp(ricker_arg)
    
    result = (params[0]                              # constant
              + params[1] * P_safe                   # linear growth
              + params[2] * P2                       # quadratic (logistic)
              + params[3] * t                        # time trend
              + params[4] * t * P_safe               # time-varying growth rate
              + params[5] * holling                  # Holling II saturation
              + params[6] * ricker                   # Ricker density dep.
              + params[8] * log_P * P_safe           # Gompertz-like P*ln(P)
              )
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
