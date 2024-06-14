import numpy as np
from scipy.stats import uniform
import warnings


def gp_fit(x, method):
    """
    Fits a generalized Pareto distribution (gPd) to a random sample using either the asymptotic maximum likelihood method (AMLE) or the combined estimation method.
    
    Parameters:
    x (numpy.ndarray): A numeric data vector containing a random sample of positive real numbers.
    method (str): The name of the parameter estimation method to be used. Options are "amle" and "combined".

    Returns:
    numpy.ndarray: A 2x1 array containing the shape and scale parameter estimates.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Data must be a numeric vector")
    if np.sum(np.isnan(x)) > 0:
        warnings.warn("NA values have been deleted")
    x = x[~np.isnan(x)]
    x = x.ravel()
    n = len(x)  # adjusted sample size without NA values
    if n <= 1:
        raise ValueError("sample size must be larger than 1")
    samplerange = np.max(x) - np.min(x)
    if samplerange == 0:
        raise ValueError("all observations are identical")
    if np.min(x) < 0:
        raise ValueError("There are negative observations. All data must be positive real numbers.")
    if method not in ["amle", "combined"]:
        raise ValueError("Unknown method. Please check the 'method' argument in the help files.")
    if method == "amle":
        fit = _amle_method(x, np.ceil(0.2 * n))
    elif method == "combined":
        fit = _combined_method(x)
    fit = np.round(fit.reshape(-1, 1), 4)
    return fit


def gp_test(x, B=999):
    """
    Bootstrap goodness-of-fit test for the Generalized Pareto distribution (gPd).

    Parameters:
    x (numpy.ndarray): A numeric data vector containing a random sample of positive real numbers.
    B (int): Number of bootstrap samples used to approximate p-values. Default is 999.

    Returns:
    dict: A dictionary containing the p-value of the test, method description, and individual p-values for H0- and H0+.
    """

    if not isinstance(x, np.ndarray):
        raise ValueError("Data must be a numeric vector")
    if np.sum(np.isnan(x)) > 0:
        warnings.warn("NA values have been deleted")
    x = x[~np.isnan(x)]
    x = x.ravel()
    n = len(x)  # sample size without NA values
    if n <= 1:
        raise ValueError("sample size must be larger than 1")
    samplerange = np.max(x) - np.min(x)
    if samplerange == 0:
        raise ValueError("all observations are identical")
    if np.min(x) < 0:
        raise ValueError("There are negative observations. All data must be positive real numbers.")
    
    gammap = _amle_method(x, np.ceil(0.2 * n))[0]
    gamman = _combined_method(x)[0]
    r1 = _R1(x)  # observed value of R^-
    r2 = _R2(x)  # observed value of R^+
    p_value1 = np.sum(np.array([_R1(_rgp(n, gamman)) < r1 for _ in range(B)]) ) / B  # boostrap p-value for H_0^-
    p_value2 = np.sum(np.array([_R2(_rgp(n, gammap)) < r2 for _ in range(B)]) ) / B  # boostrap p-value for H_0^+
    p_value = max(p_value1, p_value2)  # p-value of the intersection-union test 
    pvalues = np.round(np.array([[p_value1, r1], [p_value2, r2]]), 4)
    name1 = "H_0^-: Data follows a gPd with NEGATIVE shape parameter"
    name2 = "H_0^+: Data follows a gPd with POSITIVE shape parameter"
    results = {
        "p_value": p_value,
        "method": "Bootstrap test of fit for the generalized Pareto distribution",
        "pvalues": pvalues
    }
    return results


# INTERNAL FUNCTIONS
def _amle_method(x, k):
    """
    Asymptotic maximum likelihood estimators for the gPd.
    
    Parameters:
    x (numpy.ndarray): Sorted data vector.
    k (int): Parameter for AMLE method.

    Returns:
    numpy.ndarray: Shape and scale parameters.
    """
    x = np.sort(x)
    n = len(x)
    nk = n - k
    x1 = x[int(nk):]
    w = np.log(x1)
    g = -(w[0] - np.sum(w) / k)  # equation (4)
    sigma = g * np.exp(w[0] + g * np.log(k / n))
    return np.array([g, sigma])


def _combined_method(x):
    """
    Combined estimators for the gPd.
    
    Parameters:
    x (numpy.ndarray): Data vector.

    Returns:
    numpy.ndarray: Shape and scale parameters.
    """
    m = np.mean(x)
    maxi = np.max(x)
    g = m / (m - maxi)  # equation (7)
    sigma = -g * maxi  # equation (6)
    return np.array([g, sigma])


def _R1(x):
    """
    Test statistic for H0-

    Parameters:
    x (numpy.ndarray): Data vector.

    Returns:
    float: Test statistic R1.
    """
    gamma_neg = _combined_method(x)[0]
    Fn = np.cumsum(np.ones_like(x)) / len(x)
    x1 = x[x != np.max(x)]
    z1 = (1 - Fn[:-1]) ** (-gamma_neg)
    return np.abs(np.corrcoef(x1, z1)[0, 1])


def _R2(x):
    """
    Test statistic for H0+

    Parameters:
    x (numpy.ndarray): Data vector.

    Returns:
    float: Test statistic R2.
    """
    n = len(x)
    Fn = np.cumsum(np.ones_like(x)) / n
    gamma_positive = _amle_method(x, np.ceil(0.2 * n))[0]
    x1 = x[x != np.max(x)]
    y1 = (1 - Fn[:-1]) ** (-gamma_positive)
    x_star = np.log(x1)
    y_star = np.log(y1 - 1)
    if gamma_positive <= 0.5:
        return np.corrcoef(x1, y1)[0, 1]
    if gamma_positive > 0.5:
        return np.corrcoef(x_star, y_star)[0, 1]


def _rgp(n, shape):
    """
    Simulate random numbers from the gPd.

    Parameters:
    n (int): Number of random numbers.
    shape (float): Shape parameter for the gPd.

    Returns:
    numpy.ndarray: Random numbers from the gPd.
    """
    if shape != 0:
        return (1 / shape) * (uniform.rvs(size=n) ** (-shape) - 1)
    else:
        return np.random.exponential(1, size=n)

