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
    
    A_safe = np.abs(A) + 1e-12
    
    # Polynomial rate law: constant + first-order + second-order
    result = params[0] + params[1] * A + params[2] * A**2
    
    # General n-th order kinetics: k * A^n with safe exponent
    n = np.clip(params[4], -3.0, 5.0)
    result = result + params[3] * np.sign(A) * np.power(A_safe, n)
    
    # Logarithmic term (autocatalytic / complex mechanisms)
    result = result + params[5] * np.log(A_safe)
    
    # Time-dependent catalyst deactivation: k * A * exp(-alpha*t)
    alpha = params[7] ** 2  # ensure positive decay rate
    result = result + params[6] * A * np.exp(-np.clip(alpha * t, 0, 20))
    
    # Time terms: linear drift + concentration-time interaction
    result = result + params[8] * t + params[9] * t * A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
