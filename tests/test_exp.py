import numpy as np
import pytest
import sys
import os

# Add the goftpy directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'goftpy')))

from exp_test import exp_test


def test_exp_test_exponential():
    # Test case 1: exponential distribution with transformation test
    data = np.random.exponential(scale=1, size=250)
    result = exp_test(data, method="transf")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Test for exponentiality based on a transformation to uniformity"
    assert isinstance(result["statistic"]["T"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1

    # Test case 2: exponential distribution with ratio test
    data = np.random.exponential(scale=1, size=250)
    result = exp_test(data, method="ratio")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Cox-Oakes test for exponentiality"
    assert isinstance(result["statistic"]["CO"], float)
    assert isinstance(result["p_value"], float)
    assert 0 <= result["p_value"] <= 1


def test_exp_test_invalid_inputs():
    # Test case 1: invalid data type
    with pytest.raises(ValueError, match="Data must be a numeric vector"):
        exp_test("invalid")
    
    # Test case 2: invalid method
    with pytest.raises(ValueError, match="Invalid method"):
        exp_test([1, 2, 3], method="invalid")
    
    # Test case 3: sample size less than 2
    with pytest.raises(ValueError, match="sample size must be larger than 1"):
        exp_test([1])
    
    # Test case 4: all observations are identical
    with pytest.raises(ValueError, match="All observations are identical"):
        exp_test([1, 1, 1])
    
    # Test case 5: negative observations
    with pytest.raises(ValueError, match="The dataset contains negative observations"):
        exp_test([-1, 2, 3])


def test_exp_test_nan_values():
    # Test case 1: NaN values in the data
    data = np.random.exponential(scale=1, size=250)
    data[0] = np.nan
    with pytest.warns(UserWarning, match="NA values have been deleted"):
        result = exp_test(data, method="transf")
    assert isinstance(result, dict)
    assert "statistic" in result
    assert "p_value" in result
    assert "method" in result


if __name__ == "__main__":
    pytest.main()
