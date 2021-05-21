import numpy as np

def euclidean_distance(x1, x2):
    ## homework:start
    return __
    ## homework:end

def distance_to_data(X, x):
    """
    Compute the distance of all row vectors in X, with
    respect to x.
    
    Args:
        X (np.ndarray): Reference data, a 2D numpy array of shape
            [n_samples, n_features].
        x (np.ndarray): Test observation, a 1D numpy array.
    
    returns:
        an array of shape [n_samples, 1], which stores the distance
        of x to every row vector in X.
        
    """
    ## homework:start
    result = 
    ## homework:end
    return result

def find_nearest_neighbors_idx(X, x, k):
    """
    Find the indices of the k-nearest-neighbors of x.
    
    Args:
        X (np.ndarray): Reference observations, a 2D numpy array of
            shape [n_samples, n_features].
        x (np.array): Test observation, a 1D numpy array.
        k (int): Number of neighbors to consider
    
    returns:
        indices in X of the k-nearest neighbors of x.
        
    """
    ## homework:start
    result = 
    ## homework:end
    return result

def knn_prediction(X, y, x, k):
    """
    Find the k-nearest-neighbors of x.
    
    Args:
        X (np.ndarray): Reference observations, a 2D numpy array of
            shape [n_samples, n_features].
        y (np.array): Reference labels of shape [n_samples,]
        x (np.array): Test observation, a 1D numpy array.
        k (int): Number of neighbors to consider
    
    returns:
        indices in X of the k-nearest neighbors of x.
        
    """
    ## homework:start
    result = 
    ## homework:end
    return result

class KNN:
    def __init__(self, k=5):
        """ Stores hyperparameters of the method"""
        self.k = k
        
    def fit(self, X, y):
        """ Simply stores the training data"""
        self._X = np.copy(X)
        self._y = np.copy(y)
    
    def predict(self, X):
        """ run prediction on a batch of samples """
        iterable = (knn_prediction(self._X, self._y, x, self.k) for x in X)
        return np.fromiter(iterable, np.int32, count=X.shape[0])