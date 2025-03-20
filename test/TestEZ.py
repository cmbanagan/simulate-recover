import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from src.EZDiffFinal import EZDiffusionModel

class TestEZDiffusionModel(unittest.TestCase):
    
    def test_forward_equations(self):
        """Test if forward equations return reasonable values."""
        a, v, t = 1.0, 1.0, 0.3
        R_pred, M_pred, V_pred = forward_equations(a, v, t)
        self.assertTrue(0 < R_pred < 1, "Accuracy should be between 0 and 1")
        self.assertTrue(M_pred > 0, "Mean RT should be positive")
        self.assertTrue(V_pred > 0, "Variance should be positive")
    
    def test_inverse_equations(self):
        """Test if inverse equations correctly recover parameters within tolerance."""
        a_true, v_true, t_true = 1.5, 1.2, 0.25
        R_pred, M_pred, V_pred = forward_equations(a_true, v_true, t_true)
        R_obs, M_obs, V_obs = R_pred, M_pred, V_pred  # No noise for unit test
        a_est, v_est, t_est = inverse_equations(R_obs, M_obs, V_obs)
        self.assertAlmostEqual(a_true, a_est, delta=0.1, msg="Boundary separation incorrect")
        self.assertAlmostEqual(v_true, v_est, delta=0.1, msg="Drift rate incorrect")
        self.assertAlmostEqual(t_true, t_est, delta=0.1, msg="Nondecision time incorrect")
    
    def test_compute_errors(self):
        """Test if error computation returns correct values."""
        true_params = (1.0, 1.0, 0.3)
        est_params = (0.9, 1.1, 0.35)
        bias, squared_error = compute_errors(true_params, est_params)
        self.assertEqual(len(bias), 3)
        self.assertEqual(len(squared_error), 3)
        self.assertTrue(all(b >= 0 for b in squared_error), "Squared error should be non-negative")

    

if __name__ == '__main__':
    unittest.main()