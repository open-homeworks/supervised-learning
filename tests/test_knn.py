import unittest
import supervised_learning.knn as KNN
import numpy as np

class TestKNN(unittest.TestCase):
 
    def test_euclidean(self):
        x1 = np.array([2,3,4,5])
        x2 = np.array([9,8,7,6])
        self.assertAlmostEqual(9.165,KNN.euclidean_distance(x1, x2),2)
    
    def test_distance_to_data(self):
        X_ = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
        x_ = np.array([5, 5, 5])
        np.testing.assert_array_almost_equal([5.3851647,1.4142135, 6.164414 ], KNN.distance_to_data(X_, x_), 2)

    def test_find_nn_idx(self):
        X_ = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
        x_ = np.array([5, 5, 5])
        np.testing.assert_array_almost_equal([1, 0], KNN.find_nearest_neighbors_idx(X_, x_, 2))

    def test_prediction(self):
        X_ = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 10]])
        y_ = np.array([0,0,1])
        x_ = np.array([5, 5, 5])
        self.assertEqual(0,KNN.knn_prediction(X_, y_, x_, 2))