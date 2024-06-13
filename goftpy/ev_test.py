import numpy as np
import warnings
from scipy import stats
from itertools import combinations

def ev_test(x, dist="gumbel", method="cor", N=1000):
    # Check input validity
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
        raise ValueError("all observations are identical")
    
    if dist not in ["gumbel", "frechet", "weibull"]:
        raise ValueError("Invalid distribution. Valid distributions are 'gumbel', 'frechet', and 'weibull'.")
    
    if dist == "frechet" and np.min(x) < 0:
        raise ValueError("The dataset contains negative observations. All data must be non-negative real numbers when testing the Frechet distribution hypothesis.")
    
    if dist == "weibull" and np.max(x) > 0:
        raise ValueError("The dataset contains positive observations. All data must be negative real numbers when testing the Weibull extreme value distribution hypothesis.")
    
    if method not in ["cor", "ratio"]:
        raise ValueError("Invalid method. Valid methods are 'cor' and 'ratio'.")
    
    if dist == "frechet":
        x = np.log(x)
    if dist == "weibull":
        x = -np.log(-x)
    
    # Implement correlation test
    if method == "cor":
        x = -x
        if n < 20 or n > 250:
            raise ValueError("The correlation test requires a sample size between 20 and 250.")
    

        def cor_stat(x):
            z = np.max(list(combinations(x, 2)), axis=1)
            nz = len(z)
            Fn = np.zeros(nz)
            for i in range(nz):
                Fn[i] = np.mean(z <= z[i])
            Fn = np.clip(Fn, 1e-10, 1 - 1e-10)  # avoid log(0)
            y = np.log(-np.log(Fn))
            logic = (z != np.max(z))
            if np.sum(logic) > 1:
                r = np.corrcoef(z[logic], y[logic])[0, 1]
                rl = np.log(1 - r)
            else:
                rl = np.nan
            return rl
        
        rl = cor_stat(x)
        if np.isnan(rl):
            warnings.warn("The correlation coefficient cannot be computed due to insufficient data")
            results = {"statistic": {"R": np.nan}, "p_value": np.nan, "method": f"Correlation test of fit for the {dist} distribution"}
            return results

        median_n = -3.02921 - 0.03846 * n + 0.00023427 * n**2 - 0.000000471091 * n**3
        if n <= 60:
            s_n = 0.7588 - 0.01697 * n + 0.000399 * n**2 - 0.000003 * n**3
        else:
            s_n = 0.53
        
        p_value = stats.norm.sf(rl, loc=median_n, scale=s_n)
        r = 1 - np.exp(rl)
        results = {"statistic": {"R": r}, "p_value": p_value, "method": f"Correlation test of fit for the {dist} distribution"}
        return results


    if method == "ratio":
        def ratio_stat(x):
            m = np.mean(x)
            frac = 1 / np.arange(1, n + 1)
            summ = np.array([np.sum(frac[i:]) for i in range(n)])
            y = np.sort(x)
            s_kim = m - np.mean(y * summ)  # Kimball's estimator for the scale parameter
            t = (np.pi * s_kim)**2 / (6 * np.var(x))  # variance ratio test statistic
            return t
        
        ratio_stat_c = ratio_stat(x)  # observed value of the test statistic
        null_dist = np.array([ratio_stat(-np.log(np.random.exponential(size=n))) for _ in range(N)])
        p1 = np.sum(ratio_stat_c < null_dist) / N
        p2 = np.sum(ratio_stat_c > null_dist) / N
        p_value = 2 * min(p1, p2)  # approximated p-value by monte carlo simulation
        results = {"statistic": {"T": ratio_stat_c}, "p_value": p_value, "method": f"Variance ratio test for the {dist} distribution"}
        return results

