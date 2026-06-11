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
    
    # Physically motivated: dA/dt for reaction kinetics
    # General form: constant + linear in A + quadratic in A + power-law + time effects
    # params[0]: constant (zero-order rate)
    # params[1]: first-order rate coefficient (*A)
    # params[2]: second-order rate coefficient (*A^2)
    # params[3]: coefficient for A^n term
    # params[4]: exponent n for power-law (shifted by 0.5 to allow fractional orders)
    # params[5]: time-dependent correction coefficient
    # params[6]: time-A interaction
    # params[7]: cubic term in A
    
    A_safe = np.maximum(np.abs(A), 1e-30)
    log_A = np.log(A_safe)
    
    # Polynomial in A (zero through second order kinetics)
    result = params[0] + params[1] * A + params[2] * A**2
    
    # Michaelis-Menten saturation kinetics: V_max * A / (K_m + A)
    Km = params[3]**2 + 1e-10
    result = result + params[4] * A / (Km + A_safe)
    
    # Power-law term for fractional reaction orders
    exp_c = np.clip(params[5], -5.0, 5.0)
    result = result + params[6] * np.sign(A) * A_safe**exp_c
    
    # A*log(A) - Temkin-like kinetics
    result = result + params[7] * A * log_A
    
    # Time-A interaction with exponential decay modulation
    # Captures time-varying rate constant + catalyst deactivation
    decay_rate = params[9]**2 + 1e-10
    t_clipped = np.clip(t * decay_rate, 0, 50)
    exp_decay = np.exp(-t_clipped)
    result = result + params[8] * A * (t + exp_decay)
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
