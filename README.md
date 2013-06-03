epac
====

Embarrassingly Parallel Array Computing: EPAC is a machine learning workflow
builder.

Principles
----------

Combine Machine Learning tasks to build a workflow that may be executed in
sequential or in parallel:
- The composition of operations from the root to a leaf is a sequential pipeline.
- Internal nodes with several children (e.g.: folds of a cross-validation) lead
  to parallel execution of the children.

The execution is based on downstream/upstream data-flow paradigm:
- The top-down *downstream data-flow* is processed by Nodes from root to leaf nodes.
  Each node is identified by a unique *primary key*, it process the downstream
  data and pass it to its children. The downstream data-flow is a simple python
  dictionary. The final results are stored (*persistence*) by the leaves using a
  *intermediary key*.
- The bottom-up *upstream data-flow* start from results stored by leaves that 
  are locally reduced by nodes up to the tree's root. If no collision occurs
  in the upstream between intermediary keys results are simply moved up without
  interactions. If collision occurs, children results with similar intermediary key
  are aggregated (stacked) together. User may define a *reducer* that will be 
  applied to each (intermediary key, result) pair. Typically reducer will perform
  statistics (average over CV-folds, compute p-values over permutations) or will
  refit a model using the best arguments.


Application programing interface
--------------------------------

- `Estimator`: is the basic machine-learning building-bloc of the workflow. It is
   a user-defined object that should implements 4 methods:
  - `fit(<keyword arguments>)`: return `self`.
  - `transform(<keyword arguments>)`: is called only if the estimator is a non-leaf node.
     return an array or a dictionary. In the latter case, the returned dictionary
     is added to the downstream data-flow.
  - `predict(<keyword arguments>)`: is called only if the estimator is a leaf node. It return an 
     array or a dictionary. In the latter the returned dictionary is added to 
     results.
  - `score(<keyword arguments>)`: is called only if the estimator is a leaf node. It return an 
     scalar or a dictionary. In the latter the returned dictionary is added to 
     results.
- `Node ::= Estimator | Pipe | Methods | Grid | CV | Permutations`. The workflow
   is a tree, made of nodes of several types:
- `Pipe(Node+)`: Build pipepline with sequential execution of `Nodes`.

```python
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=10, n_features=50, n_informative=2)
# Build sequential Pipeline
# -------------------------
# 2  SelectKBest
# |
# SVM Classifier
from epac import Pipe
pipe = Pipe(SelectKBest(k=2), SVC(kernel="linear"))
pipe.fit(X=X, y=y).predict(X=X)
```

- `Methods(Node+, reducer)`: Build workflow with parallel execution of `Nodes`.
   It is the basic parallelization node. In the bottom-up results it applies the
   reducer (if provided) and results are passed up to the parrent node. It ensure
   that their are collisions between children intermediary by trying to differentiate
   them using arguments.

```python
# Multi-classifiers
# -----------------
# Methods    Methods  (Splitter)
#  /   \
# LDA  SVM      Classifiers (Estimator)
from epac import Methods
multi = Methods(LDA(),  SVC(kernel="linear"))
multi.fit(X=X, y=y)
multi.predict(X=X)
# Do both
multi.fit_predict(X=X, y=y)

# Parallelize sequential Pipeline: Anova(k best selection) + SVM.
# No collisions between upstream keys, then no aggretation.
# Methods   Methods (Splitter)
#  /   |   \
# 1    5   10  SelectKBest (Estimator)
# |    |    |
# SVM SVM SVM  Classifiers (Estimator)
anovas_svm = Methods(*[Pipe(SelectKBest(k=k), SVC(kernel="linear")) for k in 
    [1, 5, 10]])
anovas_svm.fit_predict(X=X, y=y)
anovas_svm.reduce()
```

- `Grid(Node+)`: Similar to `Methods` but Nodes should be of the same types
   and differs only with their arguments. This way collusions occur in results
   upstream leading to aggregation (stacking into grid) of results.

```python
#                   Grid                Grid (Splitter)
#                  /     \
# SVM(linear, C=1)  .... SVM(rbf, C=10) Classifiers (Estimator)
from epac import Grid
svms = Grid(*[SVC(kernel=kernel, C=C) for kernel in ("linear", "rbf") for C in [1, 10]])
svms.fit_predict(X=X, y=y)
svms.reduce()
[l.get_key() for l in svms]
[l.get_key(2) for l in svms]  # intermediary key collisions: trig aggregation
```

- `CV(Node, n_folds, y, reducer)`: Cross-validation parallelization node.

```python
# CV of LDA
# ---------
#    CV                (Splitter)
#  /   |   \
# 0    1    2  Folds      (Slicer)
# |    |    |
# LDA LDA LDA  Classifier (Estimator)
from epac import CV
from epac import SummaryStat
cv_lda = CV(LDA(), n_folds=3)
cv_lda.fit_predict(X=X, y=y)
cv_lda.reduce()
```

- `Permutations(Node, n_perms, y, permute, reducer)`:  Permutation parallelization node.

```python
# Permutations + Cross-validation
# ----------------------------------
#           Permutations           Permutations (Splitter)
#         /     |       \
#        0      1        2         Samples (Slicer)
#       |
#     CV                           CV (Splitter)
#  /   |   \
# 0    1    2                      Folds (Slicer)
# |    |    |
# LDA LDA LDA                      Classifier (Estimator)
from epac import Permutations, CV
from epac import SummaryStat, PvalPermutations
perms_cv_lda = Permutations(CV(LDA(), n_folds=3, reducer=SummaryStat()),
                       n_perms=3, permute="y")
perms_cv_lda.fit_predict(X=X, y=y)
tree.reduce()
```

- `CVBestSearchRefit(Node+, n_folds, y, reducer)`:  Cross-validation + grid-search then refit with optimals parameters.

```python
from epac import Grid, Pipe, CVBestSearchRefit
# CV + Grid search of a simple classifier
wf = CVBestSearchRefit(*[SVC(kernel="linear", C=C) for C in [.001, 1, 100]])
wf.fit_predict(X=X, y=y)
wf.reduce()

# CV + Grid search of a pipeline with a nested grid search
wf = CVBestSearchRefit(*[Pipe(SelectKBest(k=k),
                      Grid(*[SVC(kernel="linear", C=C)\
                          for C in [.0001, .001, .01, .1, 1, 10]]))
                for k in [1, 5, 10]])
wf.fit_predict(X=X, y=y)
wf.reduce()
```


Details
-------

Nodes are of three types Mapper, Splitter or Slicer.

Splitters: process downstream data-flow.
They are non leaf node  (degree >= 1) with several children.
They split the downstream data-flow to their children.
They reduce the upstream data-flow from their children, thus they are
also Reducer

Slicers: process downstream data-flow.
They are Splitters children.
They reslice the downstream data-flow.
They do nothing on the upstream data-flow.

Reducers: process upstream data-flow.
