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
    
    # Linear restoring force: -omega^2 * x
    result = params[0] * pos
    
    # Cubic nonlinearity: -beta * x^3 (Duffing-type)
    result += params[1] * pos**3
    
    # Quadratic term (asymmetric potential)
    result += params[2] * pos**2
    
    # Driving force: F * cos(omega_d * t + phi)
    phase1 = params[4] * t + params[5]
    result += params[3] * np.cos(phase1)
    
    # Sine component with SAME frequency (avoids extra nonlinear params for BFGS)
    result += params[6] * np.sin(phase1)
    
    # Quintic nonlinearity for higher-order corrections at large amplitudes
    result += params[7] * pos**5
    
    # Second harmonic response: nonlinear oscillators generate harmonics
    # cos(2*omega*t) and sin(2*omega*t) via the same phase
    phase2 = 2.0 * phase1
    result += params[8] * np.cos(phase2)
    
    # Constant offset
    result += params[9]
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
