# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:55:46 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=12, n_features=10,
                                    n_informative=2)

# Model selection using CV: CV + Grid
# -----------------------------------------
from epac import Grid, Pipe, CVBestSearchRefit, StoreMem
Xy=dict(X=X, y=y)
# CV + Grid search of a simple classifier
self = CVBestSearchRefit(*[SVC(C=C) for C in [1, 10]])
self.fit(X=X, y=y)
self.fit_predict(X=X, y=y)
wf.reduce()


# Build sequential Pipeline
# -------------------------
# 2  SelectKBest (Estimator)
# |
# SVM Classifier (Estimator)
from epac import Pipe
pipe = Pipe(SelectKBest(k=2), SVC())
pipe.fit(X=X, y=y)
pipe.predict(X=X)
pipe.fit_predict(X=X, y=y)  # Do both


# The downstream data-flow is a keyword arguments (dict) containing X and y.
# It will pass through each processing node, SelectKBest(k=2) and SVC.
# The Fit:
# Each non-leaf (here SelectKBest  node call the fit method, then apply
# the transformation on the downstream and pass it to the next node. The leaf
# node (here SVC) do not call the transformation.
# The predict:
# Similar sequential tranformation are applied on X, except that the leaf node
# call the predict method.

## Parallelization
## ===============

# Multi-classifiers
# -----------------
#         Methods       Methods (Splitter)
#        /   \
# SVM(C=1)  SVM(C=10)   Classifiers (Estimator)
from epac import Methods
multi = Methods(SVC(C=1), SVC(C=10))
multi.fit_predict(X=X, y=y)
print multi.reduce()


#        Methods          Methods (Splitter)
#          /  \
# SVM(linear)  SVM(rbf)  Classifiers (Estimator)
svms = Methods(*[SVC(kernel=kernel) for kernel in ("linear", "rbf")])
svms.fit_predict(X=X, y=y)
print svms.reduce()

# Parallelize sequential Pipeline: Anova(k best selection) + SVM.
#    Methods    Methods (Splitter)
#  /   |   \
# 1    5   10   SelectKBest (Estimator)
# |    |    |
# SVM SVM SVM   Classifiers (Estimator)
anovas_svm = Methods(*[Pipe(SelectKBest(k=k), SVC()) for k in [1, 2]])
anovas_svm.fit_predict(X=X, y=y)
print anovas_svm.reduce()


## Parallelize SVM with several parameters.
## Collisions between upstream keys, trig aggretation.
##                   Grid                Grid (Splitter)
##                  /     \
## SVM(linear, C=1)  .... SVM(rbf, C=10) Classifiers (Estimator)
## Grid and PArMethods differ onlys the way they process the upstream
## flow. With Grid Children differs only by theire arguments, and thus
## are aggregated toggether
#from epac import Grid
#svms = Grid(SVC(C=1), SVC(C=10))
#svms.fit_predict(X=X, y=y)
#svms.reduce()
#
## Two parameters
#svms = Grid(*[SVC(kernel=kernel, C=C) for kernel in ("linear", "rbf")
#    for C in [1, 10]])
#svms.fit_predict(X=X, y=y)
#svms.reduce()


# Cross-validation
# ----------------
# CV of LDA
#      CV                 (Splitter)
#  /   |   \
# 0    1    2  Folds      (Slicer)
# |    |
#   Methods               (Splitter)
#    /   \
#  LDA  SVM    Classifier (Estimator)
from epac import CV, Methods
cv = CV(Methods(LDA(), SVC(kernel="linear")))
cv.fit_predict(X=X, y=y)
print cv.reduce()


# Model selection using CV: CV + Grid
# -----------------------------------------
from epac import Grid, Pipe, CVBestSearchRefit
# CV + Grid search of a simple classifier
wf = CVBestSearchRefit(*[SVC(C=C) for C in [1, 10]])
wf.fit_predict(X=X, y=y)
wf.reduce()

# CV + Grid search of a pipeline with a nested grid search
methods = [Pipe(SelectKBest(k=k), Grid(*[SVC(C=C) for C in [1, 10]]))
                for k in [1, 5]]
wf = CVBestSearchRefit(*methods)
wf.fit_predict(X=X, y=y)
wf.reduce()


# Perms + Cross-validation of SVM(linear) and SVM(rbf) 
# -------------------------------------
#           Perms        Perm (Splitter)
#      /     |       \
#     0      1       2   Samples (Slicer)
#            |
#           CV           CV (Splitter)
#       /   |   \
#      0    1    2       Folds (Slicer)
#           |
#        Methods         Methods (Splitter)
#    /           \
# SVM(linear)  SVM(rbf)  Classifiers (Estimator)

from epac import Perms, CV, Methods
perms_cv_svm = Perms(CV(Methods(SVC(kernel="linear"), SVC(kernel="rbf"))))
perms_cv_svm.fit_predict(X=X, y=y)
perms_cv_svm.reduce()