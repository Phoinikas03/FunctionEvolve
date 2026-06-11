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
    
    A_safe = np.maximum(np.abs(A), 1e-15)
    
    # Flexible reaction order: -k * A^n
    n = np.clip(params[1], 0.01, 5.0)
    power_term = params[0] * np.sign(A) * A_safe ** n
    
    # Second distinct power-law term for mixed-order kinetics
    n2 = np.clip(params[3], 0.01, 5.0)
    power_term2 = params[2] * np.sign(A) * A_safe ** n2
    
    # Michaelis-Menten saturation: k*A / (Km + A)
    Km = np.maximum(np.abs(params[5]), 1e-10)
    mm_term = params[4] * A / (Km + A_safe)
    
    # Time-dependent rate modulation (catalyst deactivation)
    decay = np.exp(-np.clip(np.abs(params[6]) * t, 0, 20))
    time_mod = params[7] * decay * A
    
    # Hill-type cooperative term: k * A^2 / (Km^2 + A^2)
    # captures sigmoidal/cooperative kinetics (shares Km with MM term)
    Km2 = np.maximum(Km * Km, 1e-20)
    hill_term = params[8] * A * A / (Km2 + A_safe * A_safe)
    
    # Time-concentration interaction: captures time-varying rate effects
    # e.g., product inhibition accumulating over time
    tc_term = params[9] * t * A / (1.0 + np.abs(params[5]) * t)
    
    result = (power_term + power_term2 + mm_term + time_mod
              + hill_term + tc_term)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
