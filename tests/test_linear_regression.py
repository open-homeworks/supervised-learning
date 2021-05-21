import supervised_learning.linear_regression as lib
import unittest
import numpy as np

class TestLinealReg(unittest.TestCase):

    def test_prediction(self):
        value = lib.linear_prediction([[1,2]], np.array([[3], [4]]))[0,0]
        self.assertEqual(value,11)
    
    def test_val_funct(self):
        value1 = lib.mean_squared_error(np.array([1,2]), np.array([3,5]))
        value2 = lib.mean_squared_error(np.array([1,2]), np.array([1,2]))
        self.assertTupleEqual((6.5, 0.0), (value1, value2))

    def validate_gradient(self):
        value1 = lib.mean_squared_error_gradient(np.array([[1,2]]), np.array([[3]]), y_pred=np.array([[3]]))
        np.testing.assert_array_equal(value1, np.zeros((2,1)))
        value2 = lib.mean_squared_error_gradient(np.array([[0,0]]), np.array([[1]]), y_pred=np.array([[3]]))
        np.testing.assert_array_equal(value2, np.zeros((2,1)))
        value3 = lib.mean_squared_error_gradient(np.array([[2,7]]), np.array([[1]]), y_pred=np.array([[3]]))
        np.testing.assert_array_equal(value3, np.array([[4], [14]]))
    