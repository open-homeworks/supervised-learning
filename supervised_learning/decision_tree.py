import numpy as np

def _proportion(c, y):
    """ compute the proportion of data points of class c in labels. 
        y: array or series object with the data labels"""

    ## homework:start
    prop = 
    ## homework:end
    return prop

def _gini_index(y):
    """ Computes the gini index of ds """
    ## homework:start
    ## homework:end
    return gini

def _entropy(y):
    """ Computes the entropy of a data set"""
    ## homework:start
    ## homework:end
    return entropy

def _info_gain(y, y_left, y_right, impurity_fn):
    """Compute the information gain of a split."""
    ## homework:start
    ## homework:end
    return info_gain

def _split_ds(X, y, idx, value):
    """ Create left and right splits of ds based on idx and value"""
    ## homework:start
    #ds_left = Dataset(lX, ly)
    #ds_right = Dataset(rX, ry)
    ## homework:end
    return lX, ly, rX, ry


def _best_split(X, y, impurity_fn):
    """ find the best split of a dataset using the given impurity_fn"""
    #set best value to the highest possible
    ## homework:start
    best = 
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
            self._impurity_fn = lambda x: _gini_index(x)
        else:
            self._impurity_fn = lambda x: _entropy(x)
               
    
    def _best_split(self, X, y):
        return _best_split(X, y, self._impurity_fn)
    
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
      
    def fit(self, X, y):
        self._root = self._best_split(X, y)
        self._recursive_split(self._root, 1)
        
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
    
    def predict(self, X):
        if self._root is not None:
            iterable = (self._recursive_travel(self._root, xi) for xi in X)
            return np.fromiter(iterable, int, X.shape[0])
        else:
            raise RuntimeError("please train me first")