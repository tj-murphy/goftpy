import numpy as np
import warnings
from scipy import stats
from scipy.special import polygamma

def exp_test(x, method="transf", N=1000):
    """Test if a sample follows an exponential distribution"""
    if not isinstance(x, (list, np.ndarray)):
        raise ValueError("Data must be a numeric vector")
    
    x = np.array(x)
    if np.isnan(x).any():
        warnings.warn("NA values have been deleted")
        x = x[~np.isnan(x)]
    
    n = len(x)
    if n <= 1:
        raise ValueError("sample size must be larger than 1")
    
    sample_range = np.max(x) - np.min(x)
    if sample_range == 0:
        raise ValueError("All observations are identical")
    
    if np.min(x) < 0:
        raise ValueError("The dataset contains negative observations. All data must be non-negative real numbers")
    
    if method not in ["ratio", "transf"]:
        raise ValueError("Invalid method. Valid methods are 'transf' and 'ratio'")
    
    if method == "transf":
        def stat(x):
            n = len(x)
            b_check = np.cov(x, np.log(x))[0, 1]
            u = np.exp(-x / b_check)
            variance = (1 + polygamma(1, 1)) / 16 + 1 / 12 + (np.log(2) - 1) / 8
            t = np.sqrt(n / variance) * (np.mean(u) - 0.5)
            return t
        
        stat_c = stat(x)
        if n >= 200:
            p1 = stats.norm.cdf(stat_c)
            p2 = 1 - p1
            p_value = 2 * min(p1, p2)
        else:
            null_distr = np.array([stat(np.random.exponential(scale=1, size=n)) for _ in range(N)])
            p1 = np.sum(null_distr < stat_c) / N
            p2 = 1 - p1
            p_value = 2 * min(p1, p2)
        
        results = {
            "statistic": {"T": stat_c},
            "p_value": p_value,
            "method": "Test for exponentiality based on a transformation to uniformity",
        }
        return results
    
    if method == "ratio":
        def co_stat(x):
            m = np.mean(x)
            lo = np.log(x / m)
            l = np.log(x)
            v = (n + (1 / m) * np.sum(x * lo**2) - (np.sum(x * lo))**2 / (n * m**2))**(-1)
            u = n + np.sum(l) - np.sum(x * l) / m
            obs_stat = np.sqrt(v) * u
            return obs_stat
        
        stat_c = co_stat(x)
        p1 = stats.norm.cdf(stat_c)
        p2 = 1 - p1
        p_value = 2 * min(p1, p2)

        results = {
            "statistic": {"CO": stat_c},
            "p_value": p_value,
            "method": "Cox-Oakes test for exponentiality"
        }
        return results
