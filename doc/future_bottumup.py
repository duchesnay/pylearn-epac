# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:21:35 2013

@author: ed203246
"""

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
X, y = datasets.make_classification(n_samples=12, n_features=10,
                                    n_informative=2)

from epac import Methods, Pipe

self = Methods(*[Pipe(SelectKBest(k=k), SVC(kernel=kernel, C=C)) for kernel in ("linear", "rbf") for C in [1, 10] for k in [1, 2]])
self = Methods(*[Pipe(SelectKBest(k=k), SVC(C=C)) for C in [1, 10] for k in [1, 2]])


import copy
self.fit_predict(X=X, y=y)
self.reduce()
[l.get_key() for l in svms.walk_nodes()]
[l.get_key(2) for l in svms.walk_nodes()]  # intermediary key collisions: trig aggregation

"""
# Model selection using CV: CV + Grid
# -----------------------------------------
from epac import CVBestSearchRefit
# CV + Grid search of a simple classifier
wf = CVBestSearchRefit(*[SVC(C=C) for C in [1, 10]], n_folds=3)
wf.fit_predict(X=X, y=y)
wf.reduce()
"""


"""
import numpy as np
results_list = \
{'Methods/SelectKBest(k=1)/SVC(kernel=linear,C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=1)/SVC(kernel=linear,C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=1)/SVC(kernel=rbf,C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=1)/SVC(kernel=rbf,C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(kernel=linear,C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(kernel=linear,C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(kernel=rbf,C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(kernel=rbf,C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])}}

import numpy as np
run epac/utils.py
run epac/workflow/base.py

results_list=\
{'Methods/SelectKBest(k=1)/SVC(C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=1)/SVC(C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(C=1)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te': np.array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])},
 'Methods/SelectKBest(k=2)/SVC(C=10)': {'pred_te': np.array([1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]),
  'score_te': 0.83333333333333337,
  'score_tr': 0.83333333333333337,
  'true_te':np. array([1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0])}}

"""

keys_splited = [key_split(key, eval_args=True) for key in results_list.keys()]

first = keys_splited[0]
arg_grids = list() # list of [depth_idx, arg_idx, arg_name, [arg_values]]
for i in xrange(len(first)):
    if len(first[i]) > 1: # has arguments
        for arg_idx in xrange(len(first[i][1])):
            arg_grids.append([i, arg_idx, first[i][1][arg_idx][0],
                              [first[i][1][arg_idx][1]]])

# Check if Results can be stacked same depth, same node type a,d argument names
# An enumerate all possible arguments values
for other in keys_splited[1:]:
    if len(first) != len(other):
        print results_list.keys()
        raise ValueError("Results cannot be stacked: different depth")
    for i in xrange(len(first)):
        if first[i][0] != other[i][0]:
            print results_list.keys()
            raise ValueError("Results cannot be stacked: nodes have different type")
        if len(first[i]) > 1 and len(first[i][1]) != len(other[i][1]):
            print results_list.keys()
            raise ValueError("Results cannot be stacked: nodes have different length")
        if len(first[i]) > 1: # has arguments
            for arg_idx in xrange(len(first[i][1])):
                if first[i][1][arg_idx][0] != other[i][1][arg_idx][0]:
                    print results_list.keys()
                    raise ValueError("Results cannot be stacked: nodes have"
                    "argument name")
                values = [item for item in arg_grids if i==item[0] and \
                    arg_idx==item[1]][0][3]
                values.append(other[i][1][arg_idx][1])
                #values[i][1][arg_idx][1].append(other[i][1][arg_idx][1])

for grid in arg_grids:
    grid[3] = set(grid[3])

arg_grids


@classmethod
def stack_results(list_of_dict, axis_name=None,
                                   axis_values=[]):
    """Stack a list of Result(s)

    Example
    -------
   >>> _list_of_dicts_2_dict_of_lists([dict(a=1, b=2), dict(a=10, b=20)])
   {'a': [1, 10], 'b': [2, 20]}
    """

    dict_of_list = dict()
    for d in list_of_dict:
        #self.children[child_idx].signature_args
        #sub_aggregate = sub_aggregates[0]
        for key2 in d.keys():
            #key2 = sub_aggregate.keys()[0]
            result = d[key2]
            # result is a dictionary
            if isinstance(result, dict):
                if not key2 in dict_of_list.keys():
                    dict_of_list[key2] = dict()
                for key3 in result.keys():
                    if not key3 in dict_of_list[key2].keys():
                        dict_of_list[key2][key3] = ListWithMetaInfo()
                        dict_of_list[key2][key3].axis_name = axis_name
                        dict_of_list[key2][key3].axis_values = axis_values
                    dict_of_list[key2][key3].append(result[key3])
            else:  # simply concatenate
                if not key2 in dict_of_list.keys():
                    dict_of_list[key2] = ListWithMetaInfo()
                    dict_of_list[key2].axis_name = axis_name
                    dict_of_list[key2].axis_values = axis_values
                dict_of_list[key2].append(result)
    return dict_of_list