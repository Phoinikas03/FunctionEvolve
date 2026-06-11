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
    v = x[:, 2]    # velocity
    
    p2 = pos * pos
    p3 = p2 * pos
    v2 = v * v
    
    # Linear restoring force + damping
    result = params[0] * pos + params[1] * v
    
    # Duffing nonlinearity: x^3
    result += params[2] * p3
    
    # Quintic nonlinearity: x^5
    result += params[3] * p3 * p2
    
    # Van der Pol damping: x^2 * v
    result += params[4] * p2 * v
    
    # Nonlinear velocity damping: v^3
    result += params[5] * v * v2
    
    # Position-dependent quadratic velocity damping: x * v^2
    result += params[6] * pos * v2
    
    # External periodic driving (sin+cos decomposition)
    omega_t = params[7] * t
    result += params[8] * np.sin(omega_t) + params[9] * np.cos(omega_t)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
