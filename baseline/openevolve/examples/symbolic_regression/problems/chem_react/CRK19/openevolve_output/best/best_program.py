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
    
    # Physically motivated: rate law with multiple orders
    # dA/dt = c0 + c1*A + c2*A^2 + c3*A^n + c4*t*A
    # where n = params[5] is the reaction order
    
    A_safe = np.clip(np.abs(A), 1e-30, None)
    n_clipped = np.clip(params[5], -5.0, 5.0)
    A_pow_n = np.sign(A) * np.power(A_safe, n_clipped)
    
    # Time-modulated exponential decay
    exp_arg_t = np.clip(-params[6] * t, -50.0, 50.0)
    
    # Concentration-dependent saturation/inhibition
    exp_arg_A = np.clip(-params[9] * A, -50.0, 50.0)
    
    result = (params[0]                                # constant
              + params[1] * A                          # 1st order
              + params[2] * A * A                      # 2nd order
              + params[3] * A_pow_n                    # general order A^n
              + params[4] * t * A                      # time-conc coupling
              + params[7] * np.exp(exp_arg_t) * A      # exp-modulated 1st order
              + params[8] * np.exp(exp_arg_A))         # exp saturation in A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
