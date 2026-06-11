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
    
    A_safe = np.maximum(np.abs(A), 1e-30)
    sign_A = np.sign(A)
    
    # Two competing power-law rate terms (e.g., forward + reverse reactions)
    # dA/dt = p0 * A^p1 + p2 * A^p3
    n1 = np.clip(params[1], -3, 5)
    n2 = np.clip(params[3], -3, 5)
    rate1 = params[0] * sign_A * A_safe ** n1
    rate2 = params[2] * sign_A * A_safe ** n2
    
    # Polynomial corrections (constant + linear in A)
    poly = params[4] + params[5] * A + params[6] * A**2
    
    # Time-dependent rate modulation: k(t) = k0 * exp(-p8*t) type effect
    time_mod = params[7] * t * A
    
    # Logarithmic term for slow kinetics regimes
    log_term = params[8] * np.log(A_safe + np.abs(params[9]) + 1e-10)
    
    result = rate1 + rate2 + poly + time_mod + log_term
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
