import numpy as np
import sys
import os
import warnings

# Add the goftpy directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'goftpy')))

from cauchy_test import mledist, ad_stat_exp, cauchy_test

def test_mledist():
    # Test case 1: cauchy distribution
    x = np.random.standard_cauchy(500)
    loc, scale = mledist(x, dist="cauchy")
    assert np.isclose(loc, 0, atol=0.2), f"Expected location around 0, got {loc}"
    assert np.isclose(scale, 1, atol=0.2), f"Expected scale around 1, got {scale}"

    # Test case 2: unsupported distribution
    try:
        mledist(x, dist="unsupported")
        assert False, "Expected ValueError for unsupported distribution"
    except ValueError:
        pass

def test_ad_stat_exp():
    # Test case 1: cauchy distribution
    x = np.random.standard_cauchy(500)
    ad_stat = ad_stat_exp(x)
    assert ad_stat > 0, f"Expected positive Anderson-Darling statistic, got {ad_stat}"

def test_cauchy_test():
    # Test case 1: cauchy distribution with "transf" method
    x = np.random.standard_cauchy(500)
    results = cauchy_test(x, N=1000, method="transf")
    assert results["p_value"] > 0.05, f"Expected p-value > 0.05, got {results['p_value']}"

    # Test case 2: cauchy distribution with "ratio" method
    x = np.random.standard_cauchy(500)
    results = cauchy_test(x, N=1000, method="ratio")
    assert results["p_value"] > 0.05, f"Expected p-value > 0.05, got {results['p_value']}"

    # Test case 3: non-cauchy distribution
    x = np.random.standard_normal(500)
    results = cauchy_test(x, N=5000, method="transf")
    assert results["p_value"] < 0.05, f"Expected p-value < 0.05, got {results['p_value']}"

    # Test case 4: invalid input
    try:
        cauchy_test("invalid_input")
        assert False, "Expected ValueError for invalid input"
    except ValueError:
        pass

    # Test case 5: NA values
    x = np.random.standard_cauchy(500)
    x[0] = np.nan
    with warnings.catch_warnings(record=True) as w:
        results = cauchy_test(x, N=1000, method="transf")
        assert len(w) == 1, "Expected one warning message for NA values"
        assert issubclass(w[-1].category, UserWarning), "Expected UserWarning for NA values"
        assert "NA values have been deleted" in str(w[-1].message), f"Unexpected warning message: {w[-1].message}"

    # Test case 6: sample size less than 2
    x = np.random.standard_cauchy(1)
    try:
        cauchy_test(x)
        assert False, "Expected ValueError for sample size less than 2"
    except ValueError:
        pass

    # Test case 7: all observations are identical
    x = np.ones(100)
    try:
        cauchy_test(x)
        assert False, "Expected ValueError for all identical observations"
    except ValueError:
        pass

    # Test case 8: invalid method
    x = np.random.standard_cauchy(500)
    try:
        cauchy_test(x, method="invalid")
        assert False, "Expected ValueError for invalid method"
    except ValueError:
        pass

if __name__ == "__main__":
    test_mledist()
    test_ad_stat_exp()
    test_cauchy_test()
    print("All tests passed!")