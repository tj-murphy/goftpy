import numpy as np
import pytest
import sys
import os

# Add the goftpy directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'goftpy')))

from gp import gp_fit, gp_test


def test_gp_fit_valid_inputs():
    # Test case 1: valid input with "amle" method
    data = np.random.pareto(3, size=250) + 1
    result = gp_fit(data, method="amle")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)

    # Test case 2: valid input with "combined" method
    data = np.random.pareto(3, size=250) + 1
    result = gp_fit(data, method="combined")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)


def test_gp_test_valid_inputs():
    # Test case 1: valid input
    data = np.random.pareto(3, size=250) + 1
    result = gp_test(data)
    assert isinstance(result, dict)
    assert "p_value" in result
    assert "method" in result
    assert "pvalues" in result
    assert result["method"] == "Bootstrap test of fit for the generalized Pareto distribution"
    assert 0 <= result["p_value"] <= 1
    assert isinstance(result["pvalues"], np.ndarray)
    assert result["pvalues"].shape == (2, 2)


def test_gp_fit_invalid_inputs():
    # Test case 1: invalid data type
    with pytest.raises(ValueError, match="Data must be a numeric vector"):
        gp_fit("invalid", method="amle")
    
    # Test case 2: invalid method
    with pytest.raises(ValueError, match="Unknown method"):
        gp_fit(np.array([1, 2, 3]), method="invalid")
    
    # Test case 3: sample size less than 2
    with pytest.raises(ValueError, match="sample size must be larger than 1"):
        gp_fit(np.array([1]), method="amle")
    
    # Test case 4: all observations are identical
    with pytest.raises(ValueError, match="all observations are identical"):
        gp_fit(np.array([1, 1, 1]), method="amle")
    
    # Test case 5: negative observations
    with pytest.raises(ValueError, match="There are negative observations"):
        gp_fit(np.array([-1, 2, 3]), method="amle")


def test_gp_test_invalid_inputs():
    # Test case 1: invalid data type
    with pytest.raises(ValueError, match="Data must be a numeric vector"):
        gp_test("invalid")
    
    # Test case 2: sample size less than 2
    with pytest.raises(ValueError, match="sample size must be larger than 1"):
        gp_test(np.array([1]))
    
    # Test case 3: all observations are identical
    with pytest.raises(ValueError, match="all observations are identical"):
        gp_test(np.array([1, 1, 1]))
    
    # Test case 4: negative observations
    with pytest.raises(ValueError, match="There are negative observations"):
        gp_test(np.array([-1, 2, 3]))


def test_gp_fit_nan_values():
    # Test case 1: NaN values in the data
    data = np.random.pareto(3, size=250) + 1
    data[0] = np.nan
    with pytest.warns(UserWarning, match="NA values have been deleted"):
        result = gp_fit(data, method="amle")
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 1)


def test_gp_test_nan_values():
    # Test case 1: NaN values in the data
    data = np.random.pareto(3, size=250) + 1
    data[0] = np.nan
    with pytest.warns(UserWarning, match="NA values have been deleted"):
        result = gp_test(data)
    assert isinstance(result, dict)
    assert "p_value" in result
    assert "method" in result
    assert "pvalues" in result


if __name__ == "__main__":
    pytest.main()
