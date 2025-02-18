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
    if method == "cor":
        if n < 20 or n > 250:
            raise ValueError("Sample size must be between 20 and 250 for the correlation test.")

        # Generate all pairwise maxima
        pairwise_max = [max(pair) for pair in combinations(x_transformed, 2)]
        z = np.array(pairwise_max)
        nz = len(z)

        # Compute empirical CDF (FnZ)
        sorted_z = np.sort(z)
        FnZ = np.arrange(1, nz + 1) / (nz + 1)  # Empirical quantiles

        # Theoretical Weibull(1) quantiles (log(qweibull(FnZ, 1)))
        y = np.log(-np.log(1 - FnZ))  # Equivalent

        # Compute correlation (exclude maximum value)
        logic = (z != np.max(z))
        r, _ = pearsonr(z[logic], y[logic])
        rl = np.log(1 - r)  # Test statistic

        # Precomputed normal parameters
        median_n = -3.02921 - 0.03846 * n + 0.00023427 * n**2 - 4.71091e-7 * n**3
        if n<= 60:
            s_n = 0.7588 - 0.01697 * n + 0.000399 * n**2 - 3e-6 * n**3
        else:
            s_n = 0.53

        # One-tailed p-value (right-tail)
        p_value = norm.sf(rl, loc=median_n, scale=s_n)

        result = {
                "statistic": {"R": 1 - np.exp(rl)},
                "p_value": p_value,
                "method": f"Correlation test for {dist} distribution",
        }
    
    else:  # method == "ratio"
        # Compute Kimball's estimator (s.kim)
        m = np.mean(x_transformed)
        frac = 1 / np.arrange(1, n + 1)
        summ = np.array([np.sum(frac[i:]) for i in range(n)])
        y = np.sort(x_transformed)
        s_kim = m - np.mean(y * summ)

        # Variance ratio test statistic (T)
        t = (np.pi * s_kim) ** 2 / (6 * np.var(x_transformed, ddof=1))

        # Monte Carlo simulation under H0 (Gumbel)
        null_dist = []
        for _ in range(N):
            # Simulate Gumbel data: -log(rexp(n))
            sim_data = -np.log(expon.rvs(size=n))
            sim_transformed = -sim_data  # Gumbel transformation
            # Recompute Kimball's estimator
            sim_m = np.mean(sim_transformed)
            sim_summ = np.array([np.sum(frac[i:]) for i in range(n)])
            sim_y = np.sort(sim_transformed)
            sim_s_kim = sim_m - np.mean(sim_y * sim_summ)
            sim_t = (np.pi * sim_s_kim) ** 2 / (6 * np.var(sim_transformed, ddof=1))
            null_dist.append(sim_t)

        # Two-tailed p-value
        p1 = np.mean(t < null_dist)
        p2 = np.mean(t > null_dist)
        p_value = 2 * min(p1, p2)

        result = {
                "statistic": {"T": t},
                "p_value": p_value,
                "method": f"Variance ratio test for {dist} distribution",
        }

    return result
