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
    
    A2 = A * A
    A3 = A2 * A
    A_safe = np.maximum(np.abs(A), 1e-12)
    
    # Log term (common in complex/reversible kinetics)
    log_term = np.log(A_safe)
    
    # Exponential time-decay (catalyst deactivation)
    decay_rate = params[7] * params[7]
    exp_decay = np.exp(np.clip(-decay_rate * t, -50, 0))
    
    # Michaelis-Menten saturation
    K_m = params[8] * params[8] + 1e-8
    mm_term = A / (K_m + np.abs(A) + 1e-12)
    
    # Self-inhibition: A * exp(c * A)
    exp_A_arg = np.clip(params[9] * A, -50, 50)
    
    result = (params[0]
              + params[1] * A
              + params[2] * A2
              + params[3] * A3
              + params[4] * log_term
              + params[5] * t * A
              + params[6] * exp_decay * A
              + params[7] * mm_term
              + params[8] * A * np.exp(exp_A_arg))
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
