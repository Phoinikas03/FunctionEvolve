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
    
    # Physically motivated reaction kinetics model:
    # dA/dt = c0 + c1*A + c2*A^2 + c3*A^n + c4*t*A
    # where n = params[5] is the reaction order
    
    A_safe = np.maximum(np.abs(A), 1e-12)
    n = np.clip(params[4], 0.1, 5.0)
    
    # Polynomial kinetics: constant + 1st + 2nd order
    result = params[0] + params[1] * A + params[2] * A * A
    
    # Power-law term for fractional order kinetics
    result = result + params[3] * np.sign(A) * np.power(A_safe, n)
    
    # Michaelis-Menten saturation kinetics: Vmax * A / (Km + |A|)
    Km = np.abs(params[5]) + 1e-6
    result = result + params[6] * A / (Km + A_safe)
    
    # Exponential time decay modulation on first-order (catalyst deactivation)
    decay = np.exp(-np.clip(np.abs(params[7]) * t, 0, 20))
    result = result + params[8] * A * decay
    
    # Time-concentration interaction
    result = result + params[9] * t * A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
