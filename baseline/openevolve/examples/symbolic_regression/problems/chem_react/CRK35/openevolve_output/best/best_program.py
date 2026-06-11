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
    
    abs_A = np.abs(A) + 1e-12
    log_A = np.log(abs_A)
    
    # Polynomial rate law: constant + linear + quadratic in A
    result = params[0] + params[1] * A + params[2] * A**2
    
    # nth-order kinetics: k * A^n (general fractional order)
    n = np.clip(params[4], -3.0, 5.0)
    A_n = np.sign(A) * np.exp(n * log_A)
    result = result + params[3] * A_n
    
    # Autocatalytic / chemical potential term: A * ln(A)
    result = result + params[5] * A * log_A
    
    # Time-modulated exponential decay * A (first-order deactivation)
    decay_rate = params[7]**2 + 1e-8
    exp_decay = np.exp(-np.clip(decay_rate * t, 0, 20))
    result = result + params[6] * exp_decay * A
    
    # Time-modulated exponential decay * A^n (nth-order deactivation)
    decay_rate2 = params[9]**2 + 1e-8
    exp_decay2 = np.exp(-np.clip(decay_rate2 * t, 0, 20))
    result = result + params[8] * exp_decay2 * A_n
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
