import numpy as np
from scipy.stats import cauchy, anderson


def cauchy_test(x, N=1000, method="transf"):
    """
    Tests whether a dataset follows a Cauchy distribution using either:
    - Transformation to exponential data (method="transf").
    - Ratio of MLE scale to mean absolute standard deviation (method="ratio").

    Parameters:
    x (array-like): Input data.
    N (int): Number of Monte Carlo simulations for p-value approximation.
    method (str): "transf" or "ratio".

    Returns:
    dict: Test statistic, p-value, and method used.
    """

    # Convert input to numpy array and remove NaNs
    x = np.array(x)
    x = x[~np.isnan(x)]
    n = len(x)

    # Validate sample size
    if n <= 1:
        raise ValueError("Sample size must be greater than 1.")
    if method not in ["transf", "ratio"]:
        raise ValueError("Method must be 'transf' or 'ratio'.")

    # Fit Cauchy parameters to the original data
    loc, scale = cauchy.fit(x)

    if method == "transf":
        # Transform data using Cauchy CDF
        Fx = cauchy.cdf(x, loc, scale)
        # Avoid log(0) by clipping
        Fx = np.clip(Fx, 1e-10, 1 - 1e-10)
        ex = -np.log(Fx)

        # Compute observed AD statistic
        ad_stat = anderson(ex, dist="expon").statistic

        # Monte Carlo simulation
        sim_stats = []
        for _ in range(N):
            # Generate data under H0
            sim_data = cauchy.rvs(loc=loc, scale=scale, size=n)
            # Fit params to simulated data
            sim_loc, sim_scale = cauchy.fit(sim_data)
            # Transform simulated data
            Fx_sim = cauchy.cdf(sim_data, sim_loc, sim_scale)
            Fx_sim = np.clip(Fx_sim, 1e-10, 1 - 1e-10)
            ex_sim = -np.log(Fx_sim)
            # Compute AD statistic for simulated data
            sim_stat = anderson(ex_sim, dist="expon").statistic
            sim_stats.append(sim_stat)

        # Calculate p-value (with continuity correction)
        p_value = (np.sum(sim_stats >= ad_stat) + 1) / (N+1)
        return {"statistic": ad_stat, "p_value": p_value, "method": method}

    else:  # method == "ratio"
        # Compute mean absolute deviation
        mad = np.mean(np.abs(x - loc))
        ratio = scale / mad

        # Monte Carlo simulation
        sim_ratios = []
        for _ in range(N):
            # Generate data under H0
            sim_data = cauchy.rvs(loc=loc, scale=scale, size=n)
            # Fit params to simulated data
            sim_loc, sim_scale = cauchy.fit(sim_data)
            # Compute MAD for simulated data
            sim_mad = np.mean(np.abs(sim_data - sim_loc))
            sim_ratio = sim_scale / sim_mad
            sim_ratios.append(sim_ratio)

        # Calculate p-value (with continuity correction)
        p_value = (np.sum(sim_ratios >= ratio) + 1) / (N+1)
        return {"statistic": ratio, "p_value": p_value, "method": method} 

