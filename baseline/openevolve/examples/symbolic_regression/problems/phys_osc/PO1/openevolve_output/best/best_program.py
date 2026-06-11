"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dv_dt (Acceleration in Nonl-linear Harmonic Oscillator)
Input variables (columns of x): x (Position at time t), t (Time)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: x (Position at time t)
#   x[:, 1]: t (Time)

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
                        (x, t).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    pos = x[:, 0]  # position
    t = x[:, 1]    # time
    
    # Driving frequency (shared for physical consistency)
    omega = params[5]
    phase = params[6]
    
    # Linear restoring force: -alpha * x
    result = params[0] * pos
    
    # Quadratic nonlinearity (asymmetric potential)
    result += params[1] * pos**2
    
    # Cubic nonlinearity (Duffing): -beta * x^3
    result += params[2] * pos**3
    
    # Periodic driving force: F0 * cos(omega * t + phi)
    result += params[3] * np.cos(omega * t + phase)
    
    # Periodic driving sin component (captures arbitrary driving phase)
    result += params[4] * np.sin(omega * t + phase)
    
    # Damping proxy: x*sin(omega*t+phi) ~ velocity coupling
    result += params[7] * pos * np.sin(omega * t + phase)
    
    # Second harmonic driving
    result += params[8] * np.cos(2.0 * omega * t + phase)
    
    # Quintic nonlinearity
    result += params[9] * pos**5
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
