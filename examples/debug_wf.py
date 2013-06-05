# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
import sklearn.pipeline
from sklearn.cross_validation import StratifiedKFold
from sklearn import grid_search
from epac import Pipe, Methods, CV, Perms, CVBestSearchRefit
from epac import SummaryStat
from epac.sklearn_plugins import Permutations

X, y = datasets.make_classification(n_samples=12, n_features=10, n_informative=2)
n_folds_nested = 2
#random_state = 0
C_values = [.1, 1, 2, 5, 10, 100]

# With EPAC
methods = Methods(*[SVC(C=C, kernel="linear") for C in C_values])
wf = CVBestSearchRefit(methods, n_folds=n_folds_nested)
self = wf
wf.fit_predict(X=X, y=y)
r_epac = wf.reduce().values()[0]

# - Without EPAC
r_sklearn = dict()
clf = SVC(kernel="linear")
parameters = {'C': C_values}
cv_nested = StratifiedKFold(y=y, n_folds=n_folds_nested)
gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
gscv.fit(X, y)
r_sklearn['pred_te'] = gscv.predict(X)
r_sklearn['best_params'] = gscv.best_params_
print r_sklearn
print r_epac
