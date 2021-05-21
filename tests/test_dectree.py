import unittest
import supervised_learning.decision_tree as DecisionTree
import numpy as np


class TestDecisionTree(unittest.TestCase): 
    
    def test_proportion(self):
        y = np.array([0, 0, 1, 0])
        self.assertEqual(0.25, DecisionTree._proportion(1, y))
    
    def test_gini_index(self):
        y = np.array([0, 0, 1, 0])
        self.assertEqual(0.375, DecisionTree._gini_index(y))
    
    def test_entropy(self):
        y = np.array([0, 0, 1, 0])
        self.assertAlmostEqual(0.81, DecisionTree._entropy(y),2)

    def test_info_gain(self):
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_left = np.array([0, 0, 0, 1])
        y_right = np.array([0, 1, 1, 1])
        gini = DecisionTree._info_gain(y, y_left, y_right, DecisionTree._gini_index)
        entropy = DecisionTree._info_gain(y, y_left, y_right, DecisionTree._entropy)
        np.testing.assert_array_almost_equal([gini, entropy], [0.12, 0.19],2)

    def test_best_split(self):
        X_ = np.array([[1.9,1.3],
	           [2.7,1.1],
               [3.3,2.1],
               [3.6,2.9],
               [0.8,2.4],
               [9.5,3.6],
               [7.2,3.1],
               [7.4,4.3],
               [10.1,3.5],
               [6.2,3.7]])
        y_ = np.array([0,0,0,0,0,1,1,1,1,1])
        result = DecisionTree. _best_split(X_, y_, DecisionTree._gini_index)
        self.assertTupleEqual((0,3.6),(result['idx'], result['value']))
