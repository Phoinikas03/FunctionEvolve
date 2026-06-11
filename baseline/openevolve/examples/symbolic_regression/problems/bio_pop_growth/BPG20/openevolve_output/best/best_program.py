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
    
    # Generalized population growth model
    P_safe = np.clip(P, 1e-10, 1e6)
    t_safe = np.clip(t, 0, 1e6)
    
    # Logistic core: r*P + c*P^2
    logistic = params[0] * P_safe + params[1] * P_safe**2
    
    # Gompertz-like: P*log(c + P) for flexible saturation
    gompertz = params[2] * P_safe * np.log(params[3]**2 + 1.0 + P_safe)
    
    # Time-varying growth rate + time-population interaction
    time_effect = params[4] * t_safe + params[5] * t_safe * P_safe
    
    # Beverton-Holt style: P/(1 + c*P) for density dependence
    rational = params[6] * P_safe / (1.0 + params[7]**2 * P_safe + 1e-10)
    
    # Exponential transient (initial conditions effect)
    transient = params[8] * np.exp(-params[9]**2 * t_safe)
    
    result = logistic + gompertz + time_effect + rational + transient
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
