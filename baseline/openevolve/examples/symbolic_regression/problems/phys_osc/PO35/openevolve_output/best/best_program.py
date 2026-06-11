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
    pos = x[:, 0]   # position
    t = x[:, 1]     # time
    vel = x[:, 2]   # velocity
    
    p = params
    pos2 = pos * pos
    pos3 = pos2 * pos
    
    # Linear restoring force + damping
    result = p[0] * pos + p[1] * vel
    
    # Cubic nonlinearity (Duffing)
    result += p[2] * pos3
    
    # Van der Pol type nonlinear damping: x^2 * v
    result += p[3] * pos2 * vel
    
    # Periodic forcing (sin/cos decomposition - linear in params, better for BFGS)
    omega = p[4]
    wt = omega * t
    result += p[5] * np.cos(wt) + p[6] * np.sin(wt)
    
    # Quadratic velocity term (kinetic energy dependence)
    vel2 = vel * vel
    result += p[7] * vel2
    
    # Cross term: position * velocity (nonlinear coupling)
    result += p[8] * pos * vel
    
    # Cubic velocity (Rayleigh-type nonlinear damping)
    result += p[9] * vel2 * vel
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
