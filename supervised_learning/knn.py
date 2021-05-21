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
    iterable = 
    ## homework:end
    return np.fromiter(iterable, np.float32, count=X.shape[0])  

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
    # find the indices corresponding to the least distances
    ## homework:start
    indices = 
    # take the first k indices
     return __
    ## homework:end

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
    indices = 
    # select the k-nearest neighbors output
    neighbors_output = 
    # compute the output of the prediction
    cls_ = 
    ## homework:end
    return cls_

class KNN:
    def __init__(self, k=5):
        """ Stores hyperparameters of the method"""
        self.k = k
        
    def fit(self, X, y):
        """ Simply stores the training data"""
        self._X = np.copy(X)
        self._y = np.copy(y)

    def _distance_to_data(self, x):
        """
        Compute the distance of all row vectors in X, with
        respect to x.

        Args:
            x (np.ndarray): Test observation, a 1D numpy array.

        returns:
            an array of shape [n_samples, 1], which stores the distance
            of x to every row vector in X.

        """
        # we do not add euclidean_distance to the class, becasue
        # it does not require any atribute of self.
        
        iterable = (euclidean_distance(x, xi) for xi in self._X)
        return np.fromiter(iterable, np.float32, count=self._X.shape[0])    

    def _find_nearest_neighbors_idx(self, x):
        """
        Find the indices of the k-nearest-neighbors of x.

        Args:
            x (np.array): Test observation, a 1D numpy array.

        returns:
            indices in X of the k-nearest neighbors of x.

        """
        # find the indices corresponding to the least distances
        indices = np.argsort(self._distance_to_data(x))
        # take the first k indices
        return indices[:self.k]

    def _predict_one(self, x):
        """ Predict the output of x using knn """
        
        indices = self._find_nearest_neighbors_idx(x)
        # select the k-nearest neighbors output
        neighbors_output = self._y[indices]
        # compute the output of the prediction
        cls_ = np.bincount(neighbors_output).argmax()
        return cls_
    
    def predict(self, X):
        """ run prediction on a batch of samples """
        iterable = (self._predict_one(x) for x in X)
        return np.fromiter(iterable, np.int32, count=X.shape[0])