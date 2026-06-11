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
    
    # Generalized logistic with rational saturation terms
    # All terms smooth and bounded - BFGS friendly
    
    P2 = P * P
    # Use |P| for saturation scale to avoid sign issues
    Pabs2 = P2 + 1e-10
    s1 = params[6]**2 + 1e-12
    denom = 1.0 + s1 * Pabs2
    
    # Effective growth rate that varies with time
    r_eff = params[1] + params[4] * t
    
    result = (params[0]                                # constant offset
              + r_eff * P                              # time-varying linear growth
              + params[2] * P2                         # quadratic (logistic saturation)
              + params[3] * t                          # time trend in baseline
              + params[5] * P * P2 / denom             # saturating cubic
              + params[7] * P / denom                  # Allee-like / saturating linear
              + params[8] * P2 / denom                 # saturating quadratic
              + params[9] * t * P2 / denom)            # time-varying saturating quadratic
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
