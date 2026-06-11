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
    
    A_safe = np.maximum(np.abs(A), 1e-30)
    
    # Core rate law: polynomial in A (0th, 1st, 2nd, 3rd order)
    result = params[0] + params[1] * A + params[2] * A**2 + params[3] * A**3
    
    # nth-order kinetics with safe power
    n_order = np.clip(params[5], 0.1, 5.0)
    result = result + params[4] * np.sign(A) * A_safe ** n_order
    
    # Log-concentration term
    result = result + params[6] * np.log(A_safe)
    
    # Time-dependent exponential modulation on A (catalyst deactivation)
    exp_arg = np.clip(-np.abs(params[8]) * t, -20, 0)
    result = result + params[7] * np.exp(exp_arg) * A
    
    # Time-concentration cross term + sqrt(A) half-order kinetics
    result = result + params[9] * t * A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
