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
    pos = x[:, 0]
    vel = x[:, 2]
    
    p2 = pos * pos
    p3 = p2 * pos
    v2 = vel * vel
    
    # Nonlinear oscillator: sin+cos restoring force + Duffing + nonlinear damping
    # Two independent frequency parameters for sin and cos
    sin_pos = np.sin(params[4] * pos)
    cos_pos = np.cos(params[5] * pos)
    
    result = (params[0] * pos            # linear restoring force (-omega^2 * x)
            + params[1] * p3             # cubic nonlinearity (Duffing, x^3)
            + params[2] * vel            # linear damping
            + params[3] * p2 * vel       # Van der Pol type damping (x^2*v)
            + params[6] * sin_pos        # sinusoidal restoring force (odd)
            + params[7] * cos_pos        # cosine restoring force (even symmetry)
            + params[8] * pos * v2       # energy-dependent coupling (x*v^2)
            + params[9] * v2 * vel)      # cubic velocity (v^3)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
