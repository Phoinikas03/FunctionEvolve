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
    
    A_abs = np.abs(A) + 1e-30
    log_A = np.log(A_abs)
    
    # Polynomial rate law (0th, 1st, 2nd, 3rd order kinetics)
    result = params[0] + params[1] * A + params[2] * A**2 + params[3] * A**3
    
    # Saturation/Michaelis-Menten: V*A / (K + A) - enzyme kinetics
    result += params[4] * A / (np.abs(params[5]) + A_abs)
    
    # Time-modulated rate: k(t)*A where k decays exponentially
    result += params[6] * A * np.exp(-np.clip(params[7] * t, -20, 20))
    
    # Autocatalytic: A*ln(A) - captures fractional/complex order kinetics
    result += params[8] * A * log_A
    
    # Cooperative/Hill-type: A^2*ln(A) - higher order autocatalytic
    result += params[9] * A**2 * log_A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
