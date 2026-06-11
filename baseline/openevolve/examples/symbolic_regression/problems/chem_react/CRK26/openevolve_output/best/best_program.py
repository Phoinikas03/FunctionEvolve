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
    
    # Physically motivated model for reaction kinetics dA/dt
    # Polynomial in A (covers zero, first, second order kinetics)
    # Plus cross terms and nonlinear terms
    
    A_safe = np.maximum(np.abs(A), 1e-30)
    
    # Core polynomial kinetics (0th, 1st, 2nd order)
    result = params[0] + params[1] * A + params[2] * A**2
    
    # General n-th order power-law term
    n = np.clip(params[4], 0.01, 5.0)
    result = result + params[3] * np.sign(A) * A_safe**n
    
    # Michaelis-Menten saturation: Vmax*A/(Km+A)
    Km = params[5]**2 + 1e-10
    result = result + params[6] * A / (Km + A_safe)
    
    # Exponential time-modulated rate (catalyst deactivation)
    decay_rate = np.clip(params[8], -10.0, 10.0)
    result = result + params[7] * np.exp(-decay_rate * t) * A
    
    # Cubic term in A for higher-order kinetics
    result = result + params[9] * A**3
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
