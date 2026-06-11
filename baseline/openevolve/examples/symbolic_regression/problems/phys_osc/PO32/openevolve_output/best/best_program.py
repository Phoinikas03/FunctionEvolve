"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dv_dt (Acceleration in Nonl-linear Harmonic Oscillator)
Input variables (columns of x): x (Position at time t), t (Time), v (Velocity at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: x (Position at time t)
#   x[:, 1]: t (Time)
#   x[:, 2]: v (Velocity at time t)

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
                        n_features is 3.
                        If n_features is 0, x should be shape (n_samples, 0).
                        The order of columns in x must correspond to:
                        (x, t, v).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    pos = x[:, 0]  # position
    t = x[:, 1]    # time
    vel = x[:, 2]  # velocity
    
    # Nonlinear harmonic oscillator: Duffing + van der Pol + driven
    # Linear restoring force + damping + constant
    result = params[0] * pos + params[1] * vel + params[2]
    # Duffing-type nonlinearities
    result += params[3] * pos**3 + params[4] * pos**2
    # Periodic driving force with amplitude, frequency, phase
    result += params[5] * np.cos(params[6] * t + params[7])
    # Van der Pol-type nonlinear damping: pos^2 * vel
    result += params[8] * pos**2 * vel
    # Velocity cubic nonlinearity
    result += params[9] * vel**3
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
