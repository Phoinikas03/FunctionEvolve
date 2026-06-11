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
    
    P2 = P * P
    exp_arg = np.clip(-params[5] * t, -50, 50)
    safe_P = np.maximum(np.abs(P), 1e-30)
    log_P = np.log(safe_P) * np.sign(P + 1e-30)
    P2_denom = 1.0 + P2
    
    # Gompertz-logistic hybrid with cubic saturation, transient, periodic
    # params[1]*P: linear growth
    # params[2]*P^2: logistic density dependence
    # params[3]*P*log(P): Gompertz-type density dependence
    # params[4]*t*P: time-varying growth rate
    # params[6]*exp(-p5*t)*P: transient dynamics
    # params[7]*P^3/(1+P^2): cubic saturating term
    # params[9]*sin(p8*t)*P: periodic forcing
    result = (params[0]
              + params[1] * P
              + params[2] * P2
              + params[3] * P * log_P
              + params[4] * t * P
              + params[6] * np.exp(exp_arg) * P
              + params[7] * P * P2 / P2_denom
              + params[9] * np.sin(params[8] * t) * P)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
