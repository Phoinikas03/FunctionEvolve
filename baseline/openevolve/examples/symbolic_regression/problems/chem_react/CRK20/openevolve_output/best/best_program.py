"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dA_dt (Rate of change of concentration in chemistry reaction kinetics)
Input variables (columns of x): t (Time), A (Concentration at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: t (Time)
#   x[:, 1]: A (Concentration at time t)

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
                        (t, A).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    t = x[:, 0]
    A = x[:, 1]
    
    # Safe power computation handling negative A
    A_safe = np.maximum(np.abs(A), 1e-12)
    sign_A = np.sign(A)
    
    # nth-order kinetics: k * sign(A) * |A|^n
    n = np.clip(params[2], 0.01, 5.0)
    power_term = params[0] * sign_A * np.power(A_safe, n)
    
    # First order term
    linear_term = params[1] * A
    
    # Second order term
    quad_term = params[3] * A * A
    
    # Constant (source/sink)
    const_term = params[4]
    
    # Michaelis-Menten saturation: Vmax * A / (Km + |A|)
    Km = params[6] * params[6] + 1e-6
    mm_term = params[5] * A / (Km + A_safe)
    
    # Time-concentration interaction
    time_A_term = params[7] * t * A
    
    # Catalyst deactivation: exp(-k*t) modulates reaction rate
    decay_rate = params[9] * params[9]
    exp_factor = np.exp(-np.clip(decay_rate * t, 0, 20))
    
    # Time-modulated reaction rate (transient catalysis)
    exp_A_term = params[8] * exp_factor * A
    
    result = power_term + linear_term + quad_term + const_term + mm_term + time_A_term + exp_A_term
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
