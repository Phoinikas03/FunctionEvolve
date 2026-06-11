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
    
    # Core: logistic growth dP/dt = r*P*(1 - P/K) = a*P + b*P^2
    # Plus constant, time dependence, and higher-order terms
    # params[0]: constant/intercept
    # params[1]: coefficient of P (intrinsic growth rate r)
    # params[2]: coefficient of P^2 (competition term -r/K)
    # params[3]: coefficient of t (time trend)
    # params[4]: coefficient of t*P (time-varying growth rate)
    # params[5]: coefficient of P^3 (higher-order density dependence)
    # params[6]: coefficient of t^2
    # params[7]: coefficient of t^2*P
    # params[8]: coefficient of exp(-params[9]*t) seasonal/decay
    
    # Hybrid model: polynomial basis (reliable for BFGS) + nonlinear terms
    # Core logistic: params[1]*P + params[2]*P^2
    # Time effects: params[3]*t, params[4]*t*P
    # Higher order: params[5]*P^3
    # Nonlinear: ratio term P^2/(P^2 + a^2) for saturation
    # Periodic: sin(params[8]*t) for oscillatory dynamics
    
    P_safe = np.clip(np.abs(P), 1e-10, 1e10)
    logP = np.log(P_safe)
    P2 = P * P
    
    # Gompertz growth: dP/dt = r*P*(log(K) - log(P)) = r*P*log(K/P)
    # Expanded: r*P*log(K) - r*P*log(P) => params[1]*P + params[2]*P*logP
    gompertz_core = params[1] * P + params[2] * P * logP
    
    # Time-varying growth rate modification
    time_growth = params[4] * t * P + params[5] * t * P * logP
    
    # Periodic forcing on growth
    periodic = params[7] * np.sin(params[8] * t) * P
    
    # Quadratic density dependence (logistic correction)
    quadratic = params[3] * P2
    
    # Allee effect at low density
    allee_denom = np.abs(params[9]) + 1e-10
    allee = params[6] * P / (P + allee_denom)
    
    result = params[0] + gompertz_core + time_growth + periodic + quadratic + allee
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
