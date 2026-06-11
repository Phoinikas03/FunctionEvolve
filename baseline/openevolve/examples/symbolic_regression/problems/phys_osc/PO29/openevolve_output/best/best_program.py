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
    
    # Linear restoring force: -omega^2 * x
    # Cubic nonlinearity (Duffing): -alpha * x^3
    # Linear damping: -gamma * v
    # Nonlinear damping: -beta * x^2 * v
    # Possible driving force: F * cos(omega_d * t + phi)
    
    pos2 = pos * pos
    pos3 = pos2 * pos
    vel2 = vel * vel
    
    wt = params[6] * t
    cos_wt = np.cos(wt)
    sin_wt = np.sin(wt)
    
    # Driven Duffing-van der Pol oscillator
    result = (params[0] * pos              # linear restoring force
              + params[1] * vel             # linear damping
              + params[2] * pos3            # cubic nonlinearity (Duffing)
              + params[3] * (1.0 - pos2) * vel  # van der Pol: mu*(1-x^2)*v
              + params[4] * pos3 * pos2     # quintic nonlinearity
              + params[5] * cos_wt          # driving force (cos component)
              + params[7] * sin_wt          # driving force (sin component)
              + params[8] * pos * vel       # position-velocity coupling
              + params[9] * pos * vel2)     # mixed nonlinear term
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
