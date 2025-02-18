import unittest
import numpy as np
from scipy.stats import expon, norm
import sys


sys.path.insert(0, "../goftpy")
from ev_test import ev_test


class TestEvTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N = 1000


    #--------------------------------
    # Edge Cases
    #--------------------------------

    def test_small_sample_size(self):
        """Test error for sample size <= 1."""
        with self.assertRaises(ValueError):
            ev_test([5], method="cor", dist="gumbel")
        with self.assertRaises(ValueError):
            ev_test([], method="ratio", dist="gumbel")


    def test_invalid_input(self):
        """Test with invalid input (not list or numpy array)"""
        invalid_inputs = [
                "string",
                123,
                3.14,
                {"key":"value"},
                (1, 2, 3),
                True,
                None
        ]
        for inp in invalid_inputs:
            with self.assertRaises(ValueError):
                ev_test(inp)

        # Test with valid input types
        valid_inputs = [
                [1, 2, 3],
                np.array([1, 2, 3])
        ]
        for inp in valid_inputs:
            try:
                ev_test(inp, method="ratio", dist="gumbel")
            except ValueError:
                self.fail(f"Valid input {inp} raised ValueError unexpectedly.")
    

    def test_invalid_method(self):
        """Test error for invalid method."""
        data = expon.rvs(size=10)
        with self.assertRaises(ValueError):
            ev_test(data, method="invalid", dist="gumbel")


    def test_invalid_dist(self):
        """Test error for invalid distribution."""
        data = expon.rvs(size=10)
        with self.assertRaises(ValueError):
            ev_test(data, method="ratio", dist="invalid")


    def test_frechet_negative_data(self):
        """Test that negative data raises error for Fréchet test."""
        # Fréchet requires non-negative data
        data = -np.abs(np.random.exponential(size=10))
        with self.assertRaises(ValueError):
            ev_test(data, method="ratio", dist="frechet")


    def test_weibull_positive_data(self):
        """Test that positive data raises error for Weibull test."""
        # Weibull requires non-positive data
        data = np.abs(np.random.exponential(size=10))
        with self.assertRaises(ValueError):
            ev_test(data, method="ratio", dist="weibull")


    def test_identical_data(self):
        """Test handling of identical data points."""
        data = np.array([5.0, 5.0, 5.0])
        # For cor test, this will lead to an undefined correlation
        with self.assertRaises(ValueError):
            ev_test(data, method="cor", dist="gumbel")
        # For ratio test, zero variance may lead to division by zero
        with self.assertRaises(ZeroDivisionError):
            ev_test(data, method="ratio", dist="gumbel")


if __name__ == "__main__":
    unittest.main()
