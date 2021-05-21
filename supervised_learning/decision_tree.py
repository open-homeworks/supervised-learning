import numpy as np

def proportion(c, y):
    """ compute the proportion of data points of class c in labels. 
        y: array or series object with the data labels"""

    ## homework:start
    result = 
    ## homework:end
    return result

def gini_index(y):
    """ Computes the gini index of the dataset """
    ## homework:start
    result = 
    ## homework:end
    return result

def entropy(y):
    """ Computes the entropy of a data set"""
    ## homework:start
    result = 
    ## homework:end
    return result

def info_gain(y, y_left, y_right, impurity_fn):
    """Compute the information gain of a split."""
    ## homework:start
    result = 
    ## homework:end
    return result

def split_dataset(X, y, idx, value):
    """ Create left and right splits of ds based on idx and value"""
    ## homework:start
    X_left, y_left = 
    X_right, y_right = 
    ## homework:end
    return X_left, y_left, X_right, y_right


def best_split(X, y, impurity_fn):
    """ find the best split of a dataset using the given impurity_fn"""

    ## homework:start
    best_info_gain = 
    for idx in range(indices):
        for _ in _:
            ds_left, ds_right = 
            temp = 
            if ___ > ___
                best = temp
                result = {
                    'idx': idx, 
                    'value': value,
                    'ds_left': ds_left,
                    'ds_right': ds_right
                }
    ## homework:end
    return result

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, criterion='gini'):
        self._root = None
        self.max_depth = max_depth
        if criterion == 'gini':
            self._impurity_fn = gini_index
        else:
            self._impurity_fn = entropy
               
    def fit(self, X, y):
        self._root = self._best_split(X, y)
        self._recursive_split(self._root, 1)

    
    def predict(self, X):
        if self._root is not None:
            iterable = (self._recursive_travel(self._root, xi) for xi in X)
            return np.fromiter(iterable, int, X.shape[0])
        else:
            raise RuntimeError("please train me first")

    def _best_split(self, X, y):
        return best_split(X, y, self._impurity_fn)
    
    def _define_class(self, labels):
        return np.bincount(labels).argmax()
    
    def _recursive_split(self, node, depth):
        
        ds_left, ds_right = node['ds_left'], node['ds_right']
        del node['ds_left']; del node['ds_right']
        
        if len(ds_left[0]) == 0 or len(ds_right[0]) == 0:
            all_labels = np.concatenate((ds_left[1], ds_right[1]))
            val = self._define_class(all_labels)
            node['left'] = node['right'] = val
            return
            
        if depth >= self.max_depth:
            node['left'] = self._define_class(ds_left[1])
            node['right'] = self._define_class(ds_right[1])
            return 
           
        if  len(ds_left) == 1:
            node['left'] = self._define_class(ds_left[1])
        else:
            node['left'] = self._best_split(ds_left[0], ds_left[1])
            self._recursive_split(node['left'], depth + 1)
            
        if len(ds_right) == 1:
            node['right'] = self._define_class(ds_right[1])
        else:
            node['right'] = self._best_split(ds_right[0], ds_right[1])
            self._recursive_split(node['right'], depth + 1)

    def _recursive_travel(self, node, x):
        idx = node['idx']
        value = node['value']
        if x[idx] <= value:
            if isinstance(node['left'], dict):
                return self._recursive_travel(node['left'], x)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self._recursive_travel(node['right'], x)
            else:
                return node['right']
      