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
    
    # Generalized growth model combining multiple ecological mechanisms:
    # 1) Logistic core: r*P*(1 - P/K) with time-varying carrying capacity
    # 2) Saturating terms for bounded behavior
    # 3) Flexible nonlinear terms
    
    # Core logistic: params[0]*P + params[1]*P^2
    # Time-varying capacity: params[2]*P*t + params[3]*P*t^2
    # Allee/higher order: params[4]*P^2*t
    # Saturating term: params[5]*P/(P^2 + params[6]^2 + 1e-10)
    # Logarithmic growth (Gompertz-like): params[7]*P*log(|params[8]| + 1e-10 + 1/(|P| + 1e-10))
    # Constant offset: params[9]
    
    P_abs = np.abs(P) + 1e-10
    log_P = np.log(P_abs)
    
    # Learnable frequency for periodic environmental forcing
    omega = params[5]
    sin_wt = np.sin(omega * t)
    cos_wt = np.cos(omega * t)
    sin_2wt = np.sin(2.0 * omega * t)
    cos_2wt = np.cos(2.0 * omega * t)
    
    # Ecological growth model:
    # - Logistic core (P, P^2) + Gompertz (P*log(P))
    # - Time-varying growth (P*t)
    # - Learnable-frequency periodic forcing (1st & 2nd harmonics)
    # - Time-modulated seasonality for amplitude drift
    result = (params[0]
              + params[1] * P
              + params[2] * P * P
              + params[3] * P * log_P
              + params[4] * P * sin_wt
              + params[6] * P * cos_wt
              + params[7] * P * sin_2wt
              + params[8] * P * cos_2wt
              + params[9] * P * t * cos_wt)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
