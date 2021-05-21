import unittest
import supervised_learning.logistic_regression as lib
import numpy as np

class TestLogReg(unittest.TestCase):
    
    def test_sigmoid_func(self):
        self.assertEqual(lib.sigmoid(0.0), 0.5)
        assert 0.62245 <= lib.sigmoid(0.5) <= 0.62246
        assert 0.37754 <= lib.sigmoid(-0.5) <= 0.37755
        np.testing.assert_array_almost_equal(lib.sigmoid(np.array([0, 0.5, -0.5])), np.array([0.5, 0.622459, 0.377541]))
    
    def test_loss_funct(self):
        value = lib.LogisticRegression.compute_loss(np.array([1,0]), np.array([0.5, 0.5]))
        assert 0.693147 <= value <= 0.6931472
        value0 = lib.LogisticRegression.compute_loss(np.array([0, 0]), np.array([0.5, 0.5]))
        value1 = lib.LogisticRegression.compute_loss(np.array([1, 1]), np.array([0.5, 0.5]))
        self.assertEqual(value0, value1)

    def validate_gradient(self):
        value = lib.LogisticRegression.compute_gradient(np.array([[1,1], [1,1]]), np.array([[1], [1]]), np.array([[0.5], [0.5]]))
        np.testing.assert_almost_equal(value, np.array([[-0.26894142], [-0.26894142]]), 5)
    