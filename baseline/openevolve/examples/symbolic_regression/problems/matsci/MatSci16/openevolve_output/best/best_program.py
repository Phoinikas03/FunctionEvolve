"""
Initial program: A naive linear model for symbolic regression.
This model predicts the output as a linear combination of input variables
or a constant if no input variables are present.
The function is designed for vectorized input (X matrix).

Target output variable: sigma (Stress)
Input variables (columns of x): epsilon (Strain), T (Temperature)
"""
import numpy as np

# Input variable mapping for x (columns of the input matrix):
#   x[:, 0]: epsilon (Strain)
#   x[:, 1]: T (Temperature)

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
                        (epsilon, T).
        params (np.ndarray): A 1D numpy array of parameters.
                             Expected length: 10.

    Returns:
        np.ndarray: A 1D numpy array of predicted output values, shape (n_samples,).
    """
    eps = x[:, 0]
    T = x[:, 1]
    
    eps_abs = np.abs(eps) + 1e-12
    sgn = np.sign(eps)
    
    # Power-law strain hardening: A*eps + B*|eps|^n * sign(eps)
    n_exp = np.clip(params[4], 0.05, 3.0)
    strain_term = params[0] * eps + params[1] * (eps_abs ** n_exp) * sgn
    
    # Logarithmic hardening with temperature-dependent coefficient
    log_eps = np.log1p(eps_abs) * sgn
    strain_term = strain_term + params[8] * log_eps
    
    # Thermal softening: exponential decay (more physical than polynomial at high T)
    # exp(p5*T) captures Arrhenius-like thermal activation
    exp_arg = np.clip(params[5] * T, -20.0, 5.0)
    thermal_factor = np.exp(exp_arg) + params[6] * T
    
    # Combined: (strain_hardening + yield_stress) * thermal_softening
    result = (strain_term + params[2]) * thermal_factor
    
    # Additive temperature term + offset
    result = result + params[3] * T + params[7]
    
    # Nonlinear strain-temperature interaction: power-law strain * T
    # Plus log-strain * T for capturing different strain regime interactions
    result = result + params[9] * (eps_abs ** n_exp) * sgn * T
    
    return result
    
# EVOLVE-BLOCK-END

# This part remains fixed (not evolved)
def run_search():
    return func
