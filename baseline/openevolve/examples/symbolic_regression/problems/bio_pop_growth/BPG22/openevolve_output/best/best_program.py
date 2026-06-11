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
    
    # Logistic growth with time-varying rate + Gompertz correction:
    # Base: intercept + r*P + (r/K)*P^2 + c*t + d*t*P + e*t^2
    # Plus Gompertz-type P*ln(P) for better density dependence modeling
    # params[0]: intercept
    # params[1]: linear P (growth rate)
    # params[2]: P^2 (logistic carrying capacity)
    # params[3]: linear t (time trend)
    # params[4]: t*P (time-varying growth rate)
    # params[5]: t^2 (nonlinear time trend)
    # params[6]: P*ln(P) (Gompertz-type correction)
    
    P_safe = np.maximum(np.abs(P), 1e-10)
    log_P = np.log(P_safe)
    sqrt_P = np.sqrt(P_safe)
    
    # Revert to best-performing 8-param model (score 3.3867)
    # P, P^2: logistic density dependence
    # P*ln(P): Gompertz correction
    # sqrt(P): Allee effect at low populations
    # t, t*P, t^2: time dependence
    
    result = (params[0] 
              + params[1] * P 
              + params[2] * P * P 
              + params[3] * t 
              + params[4] * t * P
              + params[5] * t * t
              + params[6] * P * log_P
              + params[7] * sqrt_P)
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
