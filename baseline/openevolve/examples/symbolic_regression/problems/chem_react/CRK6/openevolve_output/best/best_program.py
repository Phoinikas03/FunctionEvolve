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
    
    # Physically motivated reaction kinetics model
    # dA/dt for general reaction mechanisms
    
    A_safe = np.abs(A) + 1e-12
    
    # Power-law rate: -k * A^n (general order reaction)
    n_order = np.clip(params[1], 0.1, 5.0)
    power_term = -params[0] * np.sign(A) * A_safe ** n_order
    
    # Linear in A (first-order decay/growth)
    linear_term = params[2] * A
    
    # Quadratic in A (second-order / dimerization)
    quad_term = params[3] * A * A
    
    # Michaelis-Menten saturation: -Vmax * A / (Km + |A|)
    Km = np.abs(params[5]) + 1e-6
    mm_term = -params[4] * A / (Km + A_safe)
    
    # Time-dependent rate modulation (catalyst deactivation)
    decay = np.exp(-np.clip(np.abs(params[7]) * t, 0, 30))
    time_dep = params[6] * A * decay
    
    # Constant source/sink
    constant_term = params[8]
    
    # Cross-term: time-concentration interaction (time-varying rate)
    cross_term = params[9] * t * A
    
    result = power_term + linear_term + quad_term + mm_term + time_dep + constant_term + cross_term
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
