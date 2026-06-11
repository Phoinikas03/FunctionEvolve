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
    
    # Ricker-logistic hybrid: combines polynomial logistic core
    # with exponential density dependence (Ricker model: P*exp(-a*P))
    # dP/dt = a0 + a1*P + a2*P^2 + a3*t + a4*t*P + a5*P*exp(a6*P)
    # The Ricker term captures density-dependent growth more parsimoniously
    # than high-order polynomials, avoiding overfitting
    
    exp_arg = np.clip(params[6] * P, -20, 20)
    
    result = (params[0] 
              + params[1] * P 
              + params[2] * P * P 
              + params[3] * t 
              + params[4] * t * P
              + params[5] * P * np.exp(exp_arg))
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
