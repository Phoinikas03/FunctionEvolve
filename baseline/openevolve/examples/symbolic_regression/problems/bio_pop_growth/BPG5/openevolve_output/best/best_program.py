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
    
    P_safe = np.clip(P, 1e-10, 1e10)
    log_P = np.log(P_safe)
    
    # Exponential time factor for transient dynamics
    exp_t = np.exp(-np.clip(params[9] * t, -20, 20))
    
    # Generalized population growth model:
    # - Gompertz core: P*log(P) captures density-dependent slowdown
    # - P*log(P)^2: asymmetric density dependence / curvature
    # - P^2: logistic quadratic density dependence
    # - sqrt(P): sub-linear growth / Allee-like effects
    # - t*P*log(P): time-varying carrying capacity
    # - exp(-at)*P: transient initial dynamics
    # - 1/log_P interaction: captures near-equilibrium behavior
    log_P2 = log_P * log_P
    
    sqrt_P = np.sqrt(P_safe)
    
    # Saturating density term: P/(1 + c*P) approaches 1/c at high P
    # This captures bounded per-capita effects better than polynomial
    denom = 1.0 + np.abs(params[8]) * P_safe * 1e-3
    P_sat = P_safe / denom
    
    result = (params[0] 
              + params[1] * P_safe 
              + params[2] * P_safe * P_safe * 1e-3
              + params[3] * P_safe * log_P
              + params[4] * P_safe * log_P2 * 1e-2
              + params[5] * t * P_safe * 1e-3
              + params[6] * t * P_safe * log_P * 1e-3
              + params[7] * exp_t * P_safe
              + params[8] * sqrt_P)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
