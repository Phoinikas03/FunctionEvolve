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
    
    A_abs = np.abs(A) + 1e-30
    A_sign = np.sign(A)
    log_A = np.log(A_abs)
    
    # Polynomial rate law (zero, first, second, third order)
    poly = params[0] + params[1] * A + params[2] * A**2 + params[3] * A**3
    
    # General order kinetics: k * |A|^n * sign(A)
    n = np.clip(params[4], -5, 5)
    power_term = params[5] * A_sign * A_abs**n
    
    # Log-concentration dependence
    log_term = params[6] * log_A
    
    # A*log(A): autocatalytic / entropy-driven kinetics
    a_log_a = params[7] * A * log_A
    
    # Time-dependent exponential decay + linear time modulation
    exp_arg = np.clip(-np.abs(params[8]) * t, -50, 50)
    time_mod = params[9] * np.exp(exp_arg) * A
    
    return poly + power_term + log_term + a_log_a + time_mod
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
