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
    absA = np.abs(A) + 1e-30
    
    # Polynomial kinetics (0th, 1st, 2nd, 3rd order)
    poly = params[0] + params[1] * A + params[2] * A**2 + params[3] * A**3
    
    # Power-law for fractional-order kinetics (smooth exponent via sigmoid)
    exp = 0.5 + 4.5 / (1.0 + np.exp(-params[4]))  # smooth mapping to (0.5, 5.0)
    power = params[5] * np.sign(A) * absA**exp
    
    # Time-modulated rate (catalyst deactivation/activation)
    decay_rate = params[6]**2 + 1e-8  # ensure positive
    time_mod = params[7] * A * np.exp(-np.clip(decay_rate * t, 0, 50))
    
    # Time-concentration cross term
    cross = params[8] * t * A
    
    # Half-order kinetics (sqrt) combined with A*log correction
    sqrt_term = params[9] * np.sign(A) * np.sqrt(absA)
    
    # Reuse poly's cubic term slot: replace A^3 with A*log(absA) for better flexibility
    result = poly + power + time_mod + cross + sqrt_term
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
