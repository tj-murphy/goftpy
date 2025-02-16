import unittest
import numpy as np
from scipy.stats import cauchy, norm, expon
import sys

sys.path.insert(0, '../goftpy')
from cauchy_test_new import cauchy_test


class TestCauchyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.N = 1000


    #----------------------------
    # Edge Cases
    #----------------------------

    def test_small_sample_size(self):
        """Test error for sample size <= 1."""
        with self.assertRaises(ValueError):
            cauchy_test([5], method="transf")
        with self.assertRaises(ValueError):
            cauchy_test([], method="ratio")

    
    #def test_identical_data(self):
     #   """Test handling of identical data points."""
      #  data = np.array([5.0, 5.0, 5.0])
       # with self.assertRaises(ValueError):
        #    cauchy_test(data, method="transf")

    
    def test_extreme_values(self):
        """Test with very large/small values."""
        data = cauchy.rvs(loc=0, scale=1, size=100) * 1e6
        result = cauchy_test(data, method="transf")
        self.assertGreater(result["p_value"], 0.05)  # Should not reject H0


    #-------------------------------
    # Method Validation
    #-------------------------------

    def test_invalid_method(self):
        """Test error for invalid method."""
        data = cauchy.rvs(size=10)
        with self.assertRaises(ValueError):
            cauchy_test(data, method="invalid_method")


    def test_method_output_keys(self):
        """Test output structure for both methods."""
        data = cauchy.rvs(size=50)
        result_transf = cauchy_test(data, method="transf")
        result_ratio = cauchy_test(data, method="ratio")
        self.assertIn("statistic", result_transf)
        self.assertIn("p_value", result_transf)
        self.assertIn("statistic", result_ratio)
        self.assertIn("p_value", result_ratio)


    #------------------------------
    # Statistical Behaviour
    #------------------------------

    def test_cauchy_data(self):
        """Test with true Cauchy data (should NOT reject H0)."""
        data = cauchy.rvs(loc=2, scale=3, size=100)
        result_transf = cauchy_test(data, method="transf", N=self.N)
        result_ratio = cauchy_test(data, method="ratio", N=self.N)
        self.assertGreater(result_transf["p_value"], 0.05)
        self.assertGreater(result_ratio["p_value"], 0.05)


    def test_normal_data(self):
        """Test with normal data (should reject H0)."""
        data = norm.rvs(loc=2, scale=3, size=100)
        result_transf = cauchy_test(data, method="transf", N=self.N)
        result_ratio = cauchy_test(data, method="ratio", N=self.N)
        self.assertLess(result_transf["p_value"], 0.05)
        self.assertLess(result_ratio["p_value"], 0.05)


    def test_exponential_data(self):
        """Test with exponential data (should reject H0)."""
        data = expon.rvs(scale=3, size=100)
        result_transf = cauchy_test(data, method="transf", N=self.N)
        result_ratio = cauchy_test(data, method="ratio", N=self.N)
        self.assertLess(result_transf["p_value"], 0.05)
        self.assertLess(result_ratio["p_value"], 0.05)


    #-----------------------------
    # Consistency Checks
    #-----------------------------

    def test_consistency_transf_method(self):
        """Test consistency of 'transf' method across runs."""
        data = cauchy.rvs(loc=0, scale=1, size=50)
        results = [
                cauchy_test(data, method="transf", N=500)["p_value"]
                for _ in range(5)
                ]
        # Allow 10% variability due to Monte Carlo randomness
        self.assertTrue(max(results) - min(results) < 0.1)


    def test_consistency_ratio_method(self):
        """Test consistency of 'ratio' method across runs."""
        data = cauchy.rvs(loc=0, scale=1, size=50)
        results = [
                cauchy_test(data, method="ratio", N=500)["p_value"]
                for _ in range(5)
                ]
        # Allow 10% variability due to Monte Carlo randomness
        self.assertTrue(max(results) - min(results) < 0.1)


if __name__ == "__main__":
    unittest.main()
