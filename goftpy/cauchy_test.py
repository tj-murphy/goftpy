import numpy as np
import warnings
from scipy import stats


def mledist(x, dist="cauchy"):
    """Maximum likelihood estimation for the Cauchy distribution"""
    if dist == "cauchy":
        params = stats.cauchy.fit(x)
        return params[0], params[1]  # return location and scale params
    else:
        raise ValueError("Unsupported distribution")
    

def ad_stat_exp(x):
    """Anderson-Darling test statistic for the exponential distribution"""
    estim = mledist(x, dist="cauchy")
    theta_hat = estim[0]
    l_hat = estim[1]
    Fx = stats.cauchy.cdf(x, theta_hat, l_hat)
    ex = -np.log(Fx)  # transformation to approx exponential data
    n = len(ex)
    b = np.mean(ex)
    s = np.sort(ex)
    theop = stats.expon.cdf(s, scale=1/b)
    ad = -n - np.sum((2 * np.arange(1, n+1) - 1) * np.log(theop) + (2 * n + 1 - 2 * np.arange(1, n+1)) * np.log(1 - theop)) / n
    return ad


def cauchy_test(x, N=1000, method="transf"):
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("x must be a numeric vector")
    
    x = np.array(x)
    if np.isnan(x).any():
        warnings.warn("NA values have been deleted")
        x = x[~np.isnan(x)]
    
    n = len(x)
    if n <= 1:
        raise ValueError("sample size must be larger than 1")
    
    sample_range = np.max(x) - np.min(x)
    if sample_range == 0:
        raise ValueError("all observations are identical")
    
    if method not in ["ratio", "transf"]:
        raise ValueError("Invalid method. Valid methods are 'transf' and 'ratio'.")
    
    if method == "transf":
        t_c = ad_stat_exp(x)
        p_value = np.sum(np.array([ad_stat_exp(np.random.standard_cauchy(n)) for _ in range(N)]) > t_c) / N  # Monte Carlo simulation
        method_str = "Test for the Cauchy distribution based on a transformation to the exponential data"
        results = {"statistic": {"T": t_c}, "p_value": p_value, "method": method_str}
        return results
    
    if method == "ratio":
        def stat(x):
            estim = mledist(x, dist="cauchy")
            theta_hat = estim[0]
            l_hat = estim[1]
            t = l_hat / np.mean(np.abs(x - theta_hat))
            return t
        
        stat_c = stat(x)
        p_value = np.sum(np.array([stat(np.random.standard_cauchy(n)) for _ in range(N)]) > stat_c) / N  # Monte Carlo simulation
        method_str = "Test for the Cauchy distribution based on the ratio of two scale estimators"
        results = {"statistic": {"T": stat_c}, "p_value": p_value, "method": method_str}
        return results
    