"""
Estimator is the basic machine-learning building-bloc of the workflow.
It is a user-defined object that should implements 4 methods:


- fit(<keyword arguments>): return self.

- transform(<keyword arguments>): is called only if the estimator is a
  non-leaf node.
  Return an array or a dictionary. In the latter case, the returned dictionary
  is added to the downstream data-flow.

- predict(<keyword arguments>): is called only if the estimator is a leaf node.
  It return an array or a dictionary. In the latter the returned dictionary is
  added to results.

- score(<keyword arguments>): is called only if the estimator is a leaf node.
  It return an scalar or a dictionary. In the latter the returned dictionary is
  added to results.


@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""

## Abreviations
## tr: train
## te: test

import re
import numpy as np
import copy
from epac.workflow.base import BaseNode, xy_split, key_push, key_split
from epac.utils import _func_get_args_names
from epac.utils import _sub_dict, _as_dict
from epac.results import ResultSet, Result
from epac.stores import StoreMem
from epac.configuration import debug
from epac.reducers import SummaryStat


## ================================= ##
## == Wrapper node for estimators == ##
## ================================= ##

class Estimator(BaseNode):
    """Node that wrap estimators"""

    def __init__(self, estimator):
        self.estimator = estimator
        super(Estimator, self).__init__()
        self._args_fit = _func_get_args_names(self.estimator.fit) \
            if hasattr(self.estimator, "fit") else None
        self._args_predict = _func_get_args_names(self.estimator.predict) \
            if hasattr(self.estimator, "predict") else None
        self._args_transform = _func_get_args_names(self.estimator.transform) \
            if hasattr(self.estimator, "transform") else None

    def get_signature(self):
        """Overload the base name method"""
        if not self.signature_args:
            return self.estimator.__class__.__name__
        else:
            args_str = ",".join([str(k) + "=" + str(self.signature_args[k])
                             for k in self.signature_args])
            args_str = "(" + args_str + ")"
            return self.estimator.__class__.__name__ + args_str

    def get_parameters(self):
        return self.estimator.__dict__

    def fit(self, recursion=True, **Xy):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="fit", recursion=recursion, **Xy)
        # Regular fit
        Xy_dict = _sub_dict(Xy, self._args_fit)
        self.estimator.fit(**Xy_dict)
        if not self.children:  # if not children compute scores
            score = self.estimator.score(**Xy_dict)
            result = Result(key=self.get_signature())
            result[Result.SCORE, Result.TRAIN] = score
            result_set = ResultSet(result)
            self.save_state(result_set, name="result_set")
        if self.children:  # transform downstream data-flow (ds) for children
            return self.transform(recursion=False, **Xy)
        else:
            return self

    def transform(self, recursion=True, **Xy):
        # transform was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="transform", recursion=recursion,
                                 **Xy)
        # Regular transform:
        # catch args_transform in ds, transform, store output in a dict
        trn_dict = _as_dict(self.estimator.transform(**_sub_dict(Xy,
                                             self._args_transform)),
                       keys=self._args_transform)
        # update ds with transformed values
        Xy.update(trn_dict)
        return Xy

    def predict(self, recursion=True, **Xy):
        # fit was called in a top-down recursive context
        if recursion:
            return self.top_down(func_name="predict", recursion=recursion,
                                 **Xy)
        if self.children:  # if children call transform
            return self.transform(recursion=False, **Xy)
        # leaf node: do the prediction
        X_dict = _sub_dict(Xy, self._args_predict)
        pred = self.estimator.predict(**X_dict)
        # load previous train result_set and store test result_set
        result = self.load_state("result_set")[self.get_signature()]
        result[Result.PRED, Result.TEST] = pred
        # If true data (args in fit but not in predict) is provided then
        # add it to result_set plus compute score
        arg_only_in_fit = set(self._args_fit).difference(set(self._args_predict))
        if arg_only_in_fit.issubset(set(Xy.keys())):
            if len(arg_only_in_fit) != 1:
                raise ValueError("Do not know how to deal with more than one "
                    "result")
            result[Result.TRUE, Result.TEST] = Xy[arg_only_in_fit.pop()]
            result[Result.SCORE, Result.TEST] = self.estimator.score(**Xy)
        return pred

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return result_set
        if not self.children:
            return self.load_state(name="result_set")
        # 1) Build sub-aggregates over children
        children_result_set = [child.reduce(store_results=False) for
            child in self.children]
        result_set = ResultSet(*children_result_set)
        # Append node signature in the keys
        for result in result_set:
            result["key"] = key_push(self.get_signature(), result["key"])
        return result_set


class CVBestSearchRefit(Estimator):
    """Cross-validation + grid-search then refit with optimals parameters.

    Average results over first axis, then find the arguments that maximize or
    minimise a "score" over other axis.

    Parameters
    ----------

    See CV parameters, plus other parameters:

    score: str
        the score name to be optimized (default "mean_score_te").

    arg_max: boolean
        True/False take parameters that maximize/minimize the score. Default
        is True.
    """

    def __init__(self, node, **kwargs):
        super(CVBestSearchRefit, self).__init__(estimator=None)
        score = kwargs.pop("score") if "score" in kwargs else "mean_score_te"
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV
        #methods = Methods(*tasks)
        cv = CV(node=node, reducer=SummaryStat(keep=False), **kwargs)
        self.score = score
        self.arg_max = arg_max
        self.add_child(cv)  # first child is the CV

    def get_signature(self):
        return self.__class__.__name__

    def get_children_top_down(self):
        """Return children during the top-down execution."""
        return []

    def fit(self, recursion=True, **Xy):
        # Fit/predict CV grid search
        cv = self.children[0]
        cv.store = StoreMem()  # local store erased at each fit
        from epac.workflow.splitters import CV
        from epac.workflow.pipeline import Pipe
        if not isinstance(cv, CV):
            raise ValueError('Child of %s is not a "CV."'
            % self.__class__.__name__)
        cv.fit_predict(recursion=True, **Xy)
        #  Pump-up results
        cv_result_set = cv.reduce(store_results=False)
        print cv_result_set
        key_val = [(result.key(), result[self.score]) for result in cv_result_set]
        mean_cv = np.asarray(zip(*key_val)[1])
        print mean_cv
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_best = np.where(mean_cv == mean_cv_opt)[0][0]
        best_key = key_val[idx_best][0]
        # Find nodes that match the best
        nodes_dict = {n.get_signature(): n for n in self.walk_true_nodes() \
            if n.get_signature() in key_split(best_key)}
        refited = Pipe(*[nodes_dict[k].estimator for k in key_split(best_key)])
        refited.store = StoreMem()    # local store erased at each fit
        self.children = self.children[:1]
        self.add_child(refited)
        refited.fit(recursion=True, **Xy)
        refited_result_set = refited.reduce(store_results=False)
        result_set = ResultSet(refited_result_set)
        result = result_set.values()[0]  # There is only one
        result["key"] = self.get_signature()
        result["best_params"] = [dict(sig) for sig in key_split(best_key, eval=True)]
        self.save_state(result_set, name="result_set")
        #to_refit.bottum_up(store_results=False)
        # Delete (eventual) about previous refit
        return self

    def predict(self, recursion=True, **Xy):
        """Call transform  with sample_set="test" """
        refited = self.children[1]
        pred = refited.predict(recursion=True, **Xy)
        # Update current results with refited prediction
        refited_result = refited.reduce(store_results=False).values()[0]
        result = self.load_state(name="result_set").values()[0]
        result.update(refited_result.payload())
        return pred

    def fit_predict(self, recursion=True, **Xy):
        Xy_train, Xy_test = xy_split(Xy)
        self.fit(recursion=False, **Xy_train)
        Xy_test = self.predict(recursion=False, **Xy_test)
        return Xy_test

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return result_set
        return self.load_state(name="result_set")
