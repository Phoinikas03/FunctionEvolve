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
    log_A = np.log(A_safe)
    
    # Power-law reaction order: p4 * sign(A) * |A|^p5
    exp_val = np.clip(params[5], 0.1, 5.0)
    power_term = params[4] * np.sign(A) * np.power(A_safe, exp_val)
    
    # Time-dependent exponential relaxation modulating A
    time_decay = params[7] * np.exp(np.clip(-params[6] * t, -20, 20))
    
    # Michaelis-Menten saturation with proper Km: p8 * A / (Km + |A|)
    Km = params[9]**2 + 1e-6
    saturation = params[8] * A / (Km + A_safe)
    
    result = (params[0]
              + params[1] * A              # first-order kinetics
              + params[2] * A * log_A      # entropy-driven kinetics (A*ln(A))
              + params[3] * A * A          # second-order kinetics
              + power_term                 # fractional-order kinetics
              + time_decay * A             # time-varying rate
              + saturation)                # saturation kinetics
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
