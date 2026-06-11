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
    
    log_A = np.log(A_safe)
    
    # Polynomial in A: constant + linear + quadratic
    result = params[0] + params[1] * A + params[2] * A**2
    
    # Time-concentration cross terms
    result += params[3] * t * A + params[4] * t * log_A
    
    # Power-law term: params[5] * A^params[6] (nth order kinetics)
    power_term = params[5] * np.sign(A) * A_safe**params[6]
    power_term = np.clip(power_term, -1e10, 1e10)
    result += power_term
    
    # Logarithmic term: naturally arises in integrated rate law derivatives
    result += params[7] * log_A
    
    # Reciprocal term: 1/A appears in second-order kinetics
    result += params[8] / A_safe * np.sign(A)
    
    # Exponential time modulation: k(t) = k0 * exp(alpha*t) * A
    exp_arg = np.clip(params[9] * t, -50, 50)
    result += params[9] * np.exp(exp_arg) * A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
