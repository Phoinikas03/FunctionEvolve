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
    
    # Generalized logistic with time-dependent growth rate
    # dP/dt = r(t)*P*(1 - P/K) + intercept
    # r(t) = params[1] + params[4]*t  (linear time-varying growth rate)
    # K via params[2]: quadratic P term
    # Plus exponential saturation term for better nonlinear fit
    # params[0]: intercept
    # params[1]: P coefficient (base growth rate)
    # params[2]: P^2 coefficient (carrying capacity)
    # params[3]: t coefficient
    # params[4]: t*P coefficient (time-varying growth)
    # params[5]: amplitude of exponential P correction
    # params[6]: rate parameter for exponential P correction
    
    P2 = P * P
    
    # Exponential saturation: captures Allee effect or other nonlinear
    # density dependence that polynomial can't capture well
    # Use params[6]^2 to ensure positive rate, clip for numerical safety
    rate = params[6] * params[6] + 1e-10
    exp_term = np.exp(-rate * P2)
    
    result = (params[0] 
              + params[1] * P 
              + params[2] * P2 
              + params[3] * t
              + params[4] * t * P
              + params[5] * P * exp_term)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
