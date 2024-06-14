import numpy as np
import pytest
import warnings
import sys
import os

# Add the goftpy directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'goftpy')))

from gamma import gamma_fit, gamma_test


def test_gamma_fit():
    # Test case 1: valid input
    data = np.random.gamma(shape=2, scale=1, size=100).astype(np.float64)
    result = gamma_fit(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result[0, 0] == "shape"
    assert result[1, 0] == "scale"
    assert np.allclose(float(result[0, 1]), 2, rtol=0.2)  # Check estimated shape parameter
    assert np.allclose(float(result[1, 1]), 1, rtol=0.2)  # Check estimated scale parameter

    # Test case 2: input with NaN values
    data_nan = np.copy(data)
    data_nan[0] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = gamma_fit(data_nan)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NA values have been deleted" in str(w[-1].message)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

    # Test case 3: invalid input type
    with pytest.raises(ValueError, match="Data must be a numeric vector"):
        gamma_fit(np.array([1, 2, 3], dtype=object))
    
    # Test case 4: sample size less than 2
    with pytest.raises(ValueError, match="sample size must be larger than 1"):
        gamma_fit(np.array([1], dtype=np.float64))

    # Test case 5: all observations are identical
    with pytest.raises(ValueError, match="All observations are identical"):
        gamma_fit(np.array([1, 1, 1], dtype=np.float64))
    
    # Test case 6: negative observations
    with pytest.raises(ValueError, match="There are negative observations"):
        gamma_fit(np.array([-1, 2, 3], dtype=np.float64))


def test_gamma_test():
    # Test case 1: valid input
    data = np.random.gamma(shape=2, scale=1, size=100).astype(np.float64)
    result = gamma_test(data)
    assert isinstance(result, dict)
    assert "statistics" in result
    assert "p_value" in result
    assert "method" in result
    assert result["method"] == "Test of fit for the Gamma distribution"
    assert isinstance(result["statistics"]["V"], float)
    assert 0 <= result["p_value"] <= 1

    # Test case 2: input with NaN values
    data_nan = np.copy(data)
    data_nan[0] = np.nan
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = gamma_test(data_nan)
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "NA values have been deleted" in str(w[-1].message)
    assert isinstance(result, dict)
    assert "statistics" in result
    assert "p_value" in result
    assert "method" in result

    # Test case 3: invalid input type
    with pytest.raises(ValueError, match="Data must be a numeric vector containing more than 1 observation"):
        gamma_test(np.array([1, 2, 3], dtype=object))
    
    # Test case 4: sample size less than 2
    with pytest.raises(ValueError, match="Data must be a numeric vector containing more than 1 observation"):
        gamma_test(np.array([1], dtype=np.float64))
    
    # Test case 5: all observations are identical
    with pytest.raises(ValueError, match="All observations are identical"):
        gamma_test(np.array([1, 1, 1], dtype=np.float64))
    
    # Test case 6: negative observations
    with pytest.raises(ValueError, match="The dataset contains negative observations"):
        gamma_test(np.array([-1, 2, 3], dtype=np.float64))


if __name__ == "__main__":
    pytest.main()
