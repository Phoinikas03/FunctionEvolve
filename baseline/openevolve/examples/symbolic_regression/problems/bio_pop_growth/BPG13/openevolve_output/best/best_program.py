"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: dP_dt (Population growth rate)
Input variables (columns of x): t (Time), P (Population at time t)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: t (Time)
#   x[:, 1]: P (Population at time t)

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
                        (t, P).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    t = x[:, 0]
    P = x[:, 1]
    P_safe = np.maximum(np.abs(P), 1e-10)
    
    log_P = np.log(P_safe)
    
    # Core logistic terms: dP/dt = a + r*P + c*P^2
    result = params[0] + params[1] * P + params[2] * P * P
    
    # Gompertz-like term: P * ln(P)
    result = result + params[3] * P * log_P
    
    # Cubic term for Allee effect / higher-order density dependence
    result = result + params[4] * P * P * P
    
    # Time-dependent modulation
    result = result + params[5] * t + params[6] * t * P
    
    # Holling type II: P / (P + half_sat)
    half_sat = params[7] * params[7] + 1e-6
    result = result + params[8] * P / (P_safe + half_sat)
    
    # P * log(P)^2 for enhanced Gompertz flexibility
    result = result + params[9] * P * log_P * log_P
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
