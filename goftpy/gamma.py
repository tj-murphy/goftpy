import numpy as np
from scipy.stats import norm
import warnings


def gamma_fit(x):
    """Estimates the parameters of a gamma distribution that best fits the given data"""
    
    if not isinstance(x, np.ndarray) or x.dtype.kind != 'f':
        raise ValueError("Data must be a numeric vector")
    if np.isnan(x).any():
        warnings.warn("NA values have been deleted")
    x = x[~np.isnan(x)]
    x = x.flatten()
    n = len(x)

    if n <= 1:
        raise ValueError("sample size must be larger than 1")
    
    samplerange = np.max(x) - np.min(x)
    if samplerange == 0:
        raise ValueError("All observations are identical")
    if np.min(x) < 0:
        raise ValueError("There are negative observations. All data must be non-negative real numbers")
    
    b_check = np.cov(x, np.log(x))[0, 1]
    a_check = np.mean(x) / b_check
    fit = np.array([["shape", a_check], ["scale", b_check]])
    return fit


def gamma_test(x):
    """Performs a goodness-of-fit test to determine whether the given data follows a gamma distribution"""

    if not isinstance(x, np.ndarray) or x.dtype.kind != 'f' or len(x) <= 1:
        raise ValueError("Data must be a numeric vector containing more than 1 observation")
    if np.isnan(x).any():
        warnings.warn("NA values have been deleted")
    x = x[~np.isnan(x)]
    x = x.flatten()

    samplerange = np.max(x) - np.min(x)
    if samplerange == 0:
        raise ValueError("All observations are identical")
    if np.min(x) < 0:
        raise ValueError("The dataset contains negative observations. All data must be non-negative real numbers")
    
    n = len(x)
    z = np.log(x)
    x_bar = np.mean(x)
    s2_x = np.var(x, ddof=1)
    b_check = np.cov(x, z)[0, 1]
    a_check = x_bar / b_check
    v = np.sqrt(n * a_check) * (s2_x / (x_bar * b_check) - 1)
    p_value = 2 * norm.sf(np.abs(v), loc=0, scale=np.sqrt(2))
    results = {
        "statistics": {"V": v},
        "p_value": p_value,
        "method": "Test of fit for the Gamma distribution"
    }
    return results
