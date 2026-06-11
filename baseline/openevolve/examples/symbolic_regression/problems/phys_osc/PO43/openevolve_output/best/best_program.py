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
    
    # Nonlinear oscillator: Duffing + Van der Pol + driving
    # Linear: restoring force + damping
    result = params[0] * pos + params[1] * vel
    # Duffing nonlinearity: x³
    result += params[2] * pos**3
    # Van der Pol damping: v·x² (amplitude-dependent)
    result += params[3] * vel * pos**2
    # Cross coupling: v·x
    result += params[4] * vel * pos
    # Nonlinear damping: v²·x (position-dependent quadratic damping)
    result += params[5] * vel**2 * pos
    # Driving force: A*sin(wt) + B*cos(wt)
    wt = params[7] * t
    sin_wt = np.sin(wt)
    cos_wt = np.cos(wt)
    result += params[6] * sin_wt + params[8] * cos_wt
    # Parametric excitation: position modulated by driving force
    result += params[9] * pos * cos_wt
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
