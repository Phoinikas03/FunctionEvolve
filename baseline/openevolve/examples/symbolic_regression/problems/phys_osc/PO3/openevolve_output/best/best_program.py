"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dv_dt (Acceleration in Nonl-linear Harmonic Oscillator)
Input variables (columns of x): t (Time), v (Velocity at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: t (Time)
#   x[:, 1]: v (Velocity at time t)

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
                        (t, v).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    t = x[:, 0]
    v = x[:, 1]
    
    # Phase for position approximation
    phase = params[4] * t + params[5]
    s = np.sin(phase)
    c = np.cos(phase)
    
    # Constant + linear damping + cubic velocity damping
    result = params[0] + params[1] * v + params[2] * v * v * v
    
    # Linear restoring force (sin + cos for arbitrary phase)
    result += params[3] * s + params[6] * c
    
    # Cubic restoring (Duffing nonlinearity): s^3 term
    result += params[7] * s * s * s
    
    # Van der Pol type: v * sin^2 (velocity-position^2 coupling)
    result += params[8] * v * s * s
    
    # Velocity-position coupling: v * sin
    result += params[9] * v * s
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
