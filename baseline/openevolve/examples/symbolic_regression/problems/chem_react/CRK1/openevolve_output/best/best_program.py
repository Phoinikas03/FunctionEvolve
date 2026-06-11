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
    
    # General order kinetics: A^n
    n = np.clip(params[5], 0.01, 5.0)
    A_power_n = np.sign(A) * np.power(A_safe, n)
    
    # Exponential time modulation (transient kinetics)
    decay_rate = np.abs(params[6]) + 1e-10
    exp_t = np.exp(-np.clip(decay_rate * t, 0, 20))
    
    # Michaelis-Menten saturation: A / (K + |A|)
    K_mm = params[9]**2 + 1e-6  # ensure positive
    mm_term = A / (K_mm + A_safe)
    
    # Generalized rate: A^n / (K2 + A) for substrate inhibition / saturation
    K2 = params[8]**2 + 1e-6
    sat_power = A_power_n / (K2 + A_safe)
    
    # Log term for Elovich-type kinetics
    log_A = np.log(A_safe)
    
    result = (params[0] 
              + params[1] * A 
              + params[2] * A**2 
              + params[3] * A_power_n
              + params[4] * t
              + params[6] * exp_t * A
              + params[7] * mm_term
              + params[8] * sat_power
              + params[9] * A * log_A)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
