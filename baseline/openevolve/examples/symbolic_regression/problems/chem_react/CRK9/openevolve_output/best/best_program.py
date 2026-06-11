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
    
    A_abs = np.abs(A) + 1e-12
    sign_A = np.sign(A)
    log_A = np.log(A_abs)
    
    # Power law: sign(A)*|A|^n for nth-order kinetics
    n = params[6]
    power_term = sign_A * np.exp(np.clip(n * log_A, -50, 50))
    
    # Second power law for competing reaction pathway
    n2 = params[8]
    power_term2 = sign_A * np.exp(np.clip(n2 * log_A, -50, 50))
    
    # Exponential time modulation
    decay = params[9]**2
    exp_term = np.exp(np.clip(-decay * t, -50, 0))
    
    # dA/dt = k0 + k1*A + k2*A^2 + k3*A^n + k4*t*A + k5*A^n2 + k7*exp(-dt)*A
    result = (params[0]
              + params[1] * A
              + params[2] * A**2
              + params[3] * power_term
              + params[4] * t * A
              + params[5] * power_term2
              + params[7] * exp_term * A)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
