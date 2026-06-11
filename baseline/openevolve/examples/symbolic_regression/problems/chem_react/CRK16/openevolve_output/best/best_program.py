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
    
    # Physically motivated: reaction kinetics often follow
    # dA/dt = c0 + c1*A + c2*A^2 + c3*A^n + time-dependent terms
    # Use polynomial in A plus power-law term
    
    A_safe = np.clip(np.abs(A), 1e-12, None)
    logA = np.log(A_safe)
    
    # Polynomial in A (0th, 1st, 2nd order kinetics)
    result = params[0] + params[1] * A + params[2] * A**2
    
    # Power-law term: A^n with optimizable exponent
    n = params[4]
    result = result + params[3] * np.sign(A) * A_safe**n
    
    # Michaelis-Menten saturation: Vmax * A / (Km + A)
    Km = params[5]**2 + 1e-8
    result = result + params[6] * A / (Km + A_safe)
    
    # A*log(A) term: autocatalytic/complex mechanisms
    result = result + params[7] * A * logA
    
    # A*log(A)^2: higher-order log correction for complex kinetics
    result = result + params[8] * A * logA**2
    
    # Time-concentration interaction: accounts for catalyst deactivation
    result = result + params[9] * t * A
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
