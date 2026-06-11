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
    A_safe = np.maximum(np.abs(A), 1e-12)
    
    # Constant term
    result = params[0]
    
    # Primary power-law rate: k1 * A^n1
    n1 = np.clip(params[2], 0.1, 5.0)
    result = result + params[1] * np.power(A_safe, n1)
    
    # Secondary power-law term (mixed-order kinetics)
    n2 = np.clip(params[4], 0.1, 5.0)
    result = result + params[3] * np.power(A_safe, n2)
    
    # Time-dependent exponential decay modulating concentration-dependent rate
    # Models catalyst deactivation: rate depends on A but rate constant decays with time
    exp_mod = np.exp(np.clip(-np.abs(params[6]) * t, -20, 20))
    result = result + params[5] * A * exp_mod
    
    # Autocatalytic/product-inhibited: A * log(A) captures nonlinear feedback
    result = result + params[7] * A * np.log(A_safe)
    
    # Substrate inhibition: Vmax * A / (Km + A + A^2/Ki)
    # More general than Michaelis-Menten, captures inhibition at high [A]
    Km = np.maximum(np.abs(params[8]), 1e-6)
    Ki = np.maximum(np.abs(params[9]), 1e-6)
    result = result + A / (Km + A_safe + A_safe * A_safe / Ki)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
