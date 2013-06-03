# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=12, n_features=10,
                                    n_informative=2)
from epac import debug
Xy = dict(X=X, y=y)
from epac import CVBestSearchRefit
# CV + Grid search of a simple classifier
self = CVBestSearchRefit(*[SVC(C=C, kernel=ker) for C in [1, 10] for ker in ["rbf", "linear"]], n_folds=2)
self.fit(X=X, y=y)
import re
import numpy as np
key2 = cv_grid_search_results.keys()[0]
result=cv_grid_search_results[key2]
cv_node=cv_grid_search
#self.fit_predict(X=X, y=y)
#self.reduce()
