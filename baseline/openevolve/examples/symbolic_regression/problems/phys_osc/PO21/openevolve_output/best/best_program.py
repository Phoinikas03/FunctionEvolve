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
    
    # Driven Duffing-Van der Pol oscillator
    # dv/dt = -w^2*x - gamma*v - beta*x^3 + mu*(1-x^2)*v + F*sin(omega*t+phi)
    omega_t = params[9] * t
    
    result = (params[0] * pos          # linear restoring force
            + params[1] * vel          # linear damping
            + params[2] * pos**3       # Duffing cubic nonlinearity
            + params[3] * pos**2       # quadratic position (asymmetric potential)
            + params[4] * pos * vel    # position-velocity coupling
            + params[5]               # constant bias
            + params[6] * pos**2 * vel # Van der Pol nonlinear damping
            + params[7] * np.sin(omega_t)  # periodic driving (sine)
            + params[8] * np.cos(omega_t)  # periodic driving (cosine)
            )
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
