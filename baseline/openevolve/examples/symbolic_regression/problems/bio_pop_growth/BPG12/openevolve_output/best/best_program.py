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
    
    # Population growth model with logistic, Gompertz, decay, and saturation terms
    P2 = P * P
    P3 = P2 * P
    logP1 = np.log1p(np.abs(P))
    
    # Time-decaying growth: captures early transient dynamics
    exp_t = np.exp(-np.clip(params[8] * t, -50, 50))
    
    # Rational time-dependent saturation: P / (1 + b*t) for late-time effects
    t_sat = P / (1.0 + np.abs(params[9]) * t + 1e-10)
    
    # Sinusoidal modulation for possible seasonal/cyclic effects
    sin_t = np.sin(params[7] * t)
    
    result = (params[0] 
              + params[1] * P 
              + params[2] * P2 
              + params[3] * t 
              + params[4] * t * P 
              + params[5] * P3
              + params[6] * P * logP1
              + sin_t * P
              + exp_t * P
              + t_sat)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
