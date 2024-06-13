import numpy as np
import pytest
import sys
import os

# Add the goftpy directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'goftpy')))

from ev_test import ev_test

def test_ev_test_gumbel():
    # Test case 1: gumbel distribution with correlation test
    data = np.random.gumbel(loc=0, scale=1, size=250)
    result = ev_test(data, dist="gumbel", method="cor")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Correlation test of fit for the gumbel distribution"
    assert isinstance(result["statistic"]["R"], float)  # correlation coefficient should be a float
    assert isinstance(result["p_value"], float)  # p-value should be a float
    assert 0 <= result["p_value"] <= 1  # p-value should be between 0 and 1

    # Test case 2: gumbel distribution with variance ratio test
    data = np.random.gumbel(loc=0, scale=1, size=250)
    result = ev_test(data, dist="gumbel", method="ratio")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Variance ratio test for the gumbel distribution"
    assert isinstance(result["statistic"]["T"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1


def test_ev_test_frechet():
    # Test case 1: frechet distribution with correlation test
    data = np.random.gamma(1, size=250)
    result = ev_test(data, dist="frechet", method="cor")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Correlation test of fit for the frechet distribution"
    assert isinstance(result["statistic"]["R"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1


def test_ev_test_weibull():
    # Test case 1: weibull distribution with correlation test
    data = -np.random.weibull(a=1, size=250)
    result = ev_test(data, dist="weibull", method="cor")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Correlation test of fit for the weibull distribution"
    assert isinstance(result["statistic"]["R"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1

    # Test case 2: weibull distribution with variance ratio test
    result = ev_test(data, dist="weibull", method="ratio")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Variance ratio test for the weibull distribution"
    assert isinstance(result["statistic"]["T"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1


def test_ev_test_invalid_inputs():
    # Test case 1: invalid distribution
    with pytest.raises(ValueError, match="Invalid distribution"):
        ev_test([1, 2, 3], dist="invalid")
    
    # Test case 2: invalid method
    with pytest.raises(ValueError, match="Invalid method"):
        ev_test([1, 2, 3], method="invalid")
    
    # Test case 3: sample size less than 2
    with pytest.raises(ValueError, match="sample size must be larger than 1"):
        ev_test([1])

    # Test case 4: all observations are identical
    with pytest.raises(ValueError, match="all observations are identical"):
        ev_test([1, 1, 1])
    
    # Test case 5: negative observations for frechet distribution
    with pytest.raises(ValueError, match="The dataset contains negative observations"):
        ev_test([-1, 2, 3], dist="frechet")
    
    # Test case 6: positive observations for weibull distribution
    with pytest.raises(ValueError, match="The dataset contains positive observations"):
        ev_test([1, -2, -3], dist="weibull")


def test_ev_test_nan_values():
    # Test case 1: NaN values in the data
    data = np.random.gumbel(loc=0, scale=1, size=250)
    data[0] = np.nan
    result = ev_test(data, dist="gumbel", method="cor")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result


if __name__ == "__main__":
    pytest.main()