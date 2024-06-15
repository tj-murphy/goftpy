import numpy as np
from scipy.stats import shapiro, norm, anderson_ksamp
import warnings


def ig_fit(x):
    """
    Maximum likelihood estimators for the Inverse Gaussian distribution based on a random sample.

    Args:
        x (array-like): A numeric data vector containing a random sample of positive real numbers.
    
    Returns:
        dict: Dictionary with the estimated parameters mu and lambda.
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number):
        raise ValueError("Data must be a numeric vector containing more than 1 observation")
    if np.ptp(x) == 0:
        raise ValueError("all observations are identical")
    if np.min(x) < 0:
        raise ValueError("The dataset contains negative observations. All data must be non-negative real numbers")
    
    n = len(x)
    mu_hat = np.mean(x)
    lambda_hat = n / np.sum(1 / x - 1 / mu_hat)

    return {"mu": mu_hat, "lambda": lambda_hat}


def gammadist_test2(z):
    """
    Anderson-Darling test for the gamma distribution with shape parameter equal to 0.5 and unknown scale parameter.

    Args:
        z (array-like): Transformed data.
    
    Returns:
        dict: Dictionary with the test statistic and p-value.
    """
    n = len(z)
    b = 2 * np.mean(z)
    s = np.sort(z) / b
    theop = anderson_ksamp([s, np.random.gamma(0.5, size=n)])[0]
    ad = theop
    m = 0.6655
    l = 1.6
    p_value = 1 - (norm.cdf(np.sqrt(1 / ad) * (ad / m - 1)) + np.exp(2 * 1 / m) * norm.cdf(-np.sqrt(1 / ad) * (ad / m + 1)))
    results = {"AD": ad, "p_value": p_value}
    return results


def ig_test(x, method="transf"):
    """
    Implementation of tests of fit for Inverse Gaussian distributions with unknown parameters.

    Args:
        x (array-like): A numeric data vector containing a random sample of positive real numbers.
        method (str): Character string giving the name of the method to be used for testing the Inverse Gaussian hypothesis.
                      Two available options are "transf" and "ratio". Default is "transf".
    
    Returns:
        A dictionary with class "htest" containing the test results.
    """
    x = np.asarray(x)
    if not np.issubdtype(x.dtype, np.number) or len(x) <= 1:
        raise ValueError("Data must be a numeric vector containing more than 1 observation")
    if np.isnan(x).any():
        warnings.warn("NA values have been deleted")
        x = x[~np.isnan(x)]
    if np.ptp(x) == 0:
        raise ValueError("all observations are identical")
    n = len(x)
    if np.min(x) < 0:
        raise ValueError("The dataset contains negative observations. All data must be non-negative real numbers.")
    if method not in ["ratio", "transf"]:
        raise ValueError("Invalid method. Valid methods are 'transf' and 'ratio'. ")
    
    if method == "transf":
        # Test based on a transformation to normality
        u = np.random.binomial(1, 0.5, n)
        y = np.abs((x - np.mean(x)) / np.sqrt(x))
        result = shapiro(y * (1 - u) + y * (-u))
        results1 = {
            "statistic": result.statistic,
            "p_value": result.pvalue,
            "method": "Test for Inverse Gaussian distributions using a transformation to normality"
        }

        # Test based on a transformation to gamma variables
        z = ((x - np.mean(x))**2) / x
        res = gammadist_test2(z)
        results2 = {
            "statistic": res["AD"],
            "p_value": res["p_value"],
            "method": "Test for Inverse Gaussian distributions using a transformation to gamma variables",
            "alternative": "Data does not follow an Inverse Gaussian distribution."
        }
        return [results1, results2]
    
    if method == "ratio":
        m = np.mean(x)
        v = np.mean(1 / x - 1 / m)
        s2 = np.var(x, ddof=1)
        r = s2 / (m**3 * v)
        t = np.sqrt(n / (6 * m * v)) * (r -1)
        p_value = 2 * norm.sf(np.abs(t))
        results = {
            "statistic": t,
            "p_value": p_value,
            "method": "Variance ratio test for Inverse Gaussian distributions",
            "alternative": "Data does not follow an Inverse Gaussian distribution."
        }
        return results


# Example usage of ig_fit
np.random.seed(0)
x = np.random.gamma(10, 1, 50)
fit = ig_fit(x)
print("Inverse Gaussian MLE Fit:")
print(fit)

# Example usage of ig_test using "transf" method
y = np.random.lognormal(mean=0, sigma=1, size=500)
test_results_transf = ig_test(y, method="transf")
print("\nInverse Gaussian Test (Transformation Method):")
print(test_results_transf)

# Example usage of ig_test using "ratio" method
test_results_ratio = ig_test(y, method="ratio")
print("\nInverse Gaussian Test (Variance Ratio Method):")
print(test_results_ratio)