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
    
    # Linear damping: -delta * v
    linear = params[0] * v
    
    # Cubic velocity term (nonlinear damping)
    cubic = params[1] * v**3
    
    # Fundamental frequency phase
    phase1 = params[3] * t + params[4]
    
    # Restoring force (position ~ sin(wt+phi), fundamental)
    sin_term = params[2] * np.sin(phase1)
    
    # Cosine at fundamental (allows arbitrary phase of restoring force)
    cos_term = params[5] * np.cos(phase1)
    
    # Third harmonic with independent phase (from cubic nonlinearity x^3)
    sin3 = params[6] * np.sin(3.0 * params[3] * t + params[7])
    
    # Velocity-position coupling: v*cos(wt) captures v*x interaction
    v_cos = params[8] * v * np.cos(phase1)
    
    # Velocity-position² coupling: v*cos(2wt) captures v*x² interaction
    # (van der Pol: (1-x²)*v → v - v*x² where x²~(1-cos(2wt))/2)
    v_cos2 = params[9] * v * np.cos(2.0 * phase1)
    
    result = linear + cubic + sin_term + cos_term + sin3 + v_cos + v_cos2
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
