import numpy as np
from scipy.stats import pearsonr, norm, expon
from itertools import combinations


def ev_test(x, dist="gumbel", method="cor", N=1000):
    """
    Tests whether a dataset follows a Gumbel, Fréchet, or Weibull distribution.

    Parameters:
    x (array-like): Input data (numeric list or numpy array).
    dist (str): Distribution to test ("gumbel", "frechet", "weibull") Default:
    "gumbel".
    method (str): Test method ("cor" for correlation, "ratio" for variance
    ratio). Default: "cor".

    N (int): Number of Monte Carlo samples for "ratio" method. Default: 1000.

    Returns:
    dict: Test statistic, p-value, method, and data name.
    """

    # Validate input
    x = np.array(x)
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("Data must be a list or numpy array.")
    x = x[~np.isnan(x)]
    n = len(x)
    if n <= 1:
        raise ValueError("Sample size must be > 1.")
    if dist not in ["gumbel", "frechet", "weibull"]:
        raise ValueError("Invalid distribution. Use 'gumbel', 'frechet', or 'weibull'.")
    if method not in ["cor", "ratio"]:
        raise ValueError("Method must be 'cor' or 'ratio'.")

    # Distribution-specific checks
    if dist == "frechet" and np.any(x < 0):
        raise ValueError("Fréchet test requires non-negative data.")
    if dist == "weibull" and np.any(x > 0):
        raise ValueError("Weibull test requires non-positive data.")

    # Transform data based on distribution
    if dist == "frechet":
        x_transformed = np.log(x)
    elif dist == "weibull":
        x_transformed = -np.log(-x)
    else:  # Gumbel
        x_transformed = -x.copy()  # Copy to avoid modifying og data

    # Correlation test

