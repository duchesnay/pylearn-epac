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
        self.signature2_args_str = None
        super(Estimator, self).__init__()
        self._args_fit = _func_get_args_names(self.estimator.fit) \
            if hasattr(self.estimator, "fit") else None
        self._args_predict = _func_get_args_names(self.estimator.predict) \
            if hasattr(self.estimator, "predict") else None
        self._args_transform = _func_get_args_names(self.estimator.transform) \
            if hasattr(self.estimator, "transform") else None

    def get_signature(self, nb=1):
        """Overload the base name method.
        - Use estimator.__class__.__name__
        - If signature2_args_str is not None, use it. This way we have
        intermediary keys collision which trig aggregation."""
        if not self.signature_args:
            return self.estimator.__class__.__name__
        elif nb is 2 and self.signature2_args_str:
            return self.estimator.__class__.__name__ + "(" + \
                    self.signature2_args_str + ")"
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
            results = ResultSet()
            results.add(result)
            self.save_state(results, name="results")
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
        # load previous train results and store test results
        result = self.load_state("results")[self.get_signature()]
        result[Result.PRED, Result.TEST] = pred
        # If true data (args in fit but not in predict) is provided then
        # add it to results plus compute score
        arg_only_in_fit = set(self._args_fit).difference(set(self._args_predict))
        if arg_only_in_fit.issubset(set(Xy.keys())):
            if len(arg_only_in_fit) != 1:
                raise ValueError("Do not know how to deal with more than one "
                    "result")
            result[Result.TRUE, Result.TEST] = Xy[arg_only_in_fit.pop()]
            result[Result.SCORE, Result.TEST] = self.estimator.score(**Xy)
        return pred

    def reduce(self, store_results=True):
        # Terminaison (leaf) node return results
        if not self.children:
            return self.load_state(name="results")
        # 1) Build sub-aggregates over children
        children_results = [child.reduce(store_results=False) for
            child in self.children]
        results = ResultSet(children_results)
        # Append node signature in the keys
        for result in results:
            result["key"] = key_push(self.get_signature(), result["key"])
        return results


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

    def __init__(self, *tasks, **kwargs):
        super(CVBestSearchRefit, self).__init__(estimator=None)
        score = kwargs.pop("score") if "score" in kwargs else "mean_score_te"
        arg_max = kwargs.pop("arg_max") if "arg_max" in kwargs else True
        from epac.workflow.splitters import CV, Grid
        grid = Grid(*tasks)
        cv = CV(node=grid, reducer=SummaryStat(keep=False), **kwargs)
        self.score = score
        self.arg_max = arg_max
        self.add_child(cv)  # first child is the CV

    def get_signature(self, nb=1):
        return self.__class__.__name__

    def get_children_top_down(self):
        """Return children during the top-down execution."""
        return []

    def get_children_bottum_up(self):
        """Return children during the bottum-up execution."""
        return []

    def fit(self, recursion=True, **Xy):
        # Fit/predict CV grid search
        cv = self.children[0]
        cv.store = StoreMem()  # local store erased at each fit
        from epac.workflow.splitters import CV
        if not isinstance(cv, CV):
            raise ValueError('Child of %s is not a "CV."'
            % self.__class__.__name__)
        cv.fit_predict(recursion=True, **Xy)
        #  Pump-up results
        cv_result_set = cv.reduce(store_results=True)
        key_val = [(result.key(), result[self.score]) for result in cv_result_set]
        mean_cv = np.asarray(zip(*key_val)[1])
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_best = np.where(mean_cv == mean_cv_opt)[0][0]
        best_signature = key_val[idx_best][0]
        pipeline = Pipe(*[eval(m) for m in key_split(best_signature)])
        
#        
#        if debug.DEBUG:
#            debug.current = self
#            debug.Xy = Xy
#        cv_grid_search_results = cv_grid_search.load_state(name="results")
#        cv_grid_search.store = None
#        for key2 in cv_grid_search_results:
#            pipeline = self.cv_grid_search(key2=key2,
#                                           result=cv_grid_search_results[key2],
#                                           cv_node=cv_grid_search)
#            methods.append(pipeline)
#        # Add children
#        from epac.workflow.splitters import Methods
#        to_refit = Methods(*methods)
        pipeline.store = StoreMem()    # local store erased at each fit
        self.children = self.children[:1]
        self.add_child(pipeline)
        pipeline.fit(recursion=True, **Xy)
        #to_refit.bottum_up(store_results=False)
        # Delete (eventual) about previous refit
        return self

    def predict(self, recursion=True, **Xy):
        """Call transform  with sample_set="test" """
        refited = self.children[1]
        pred = refited.predict(recursion=True, **Xy)
        res = refited.bottum_up(store_results=False)
        self.save_state(res, name="results")
        return pred

    def fit_predict(self, recursion=True, **Xy):
        Xy_train, Xy_test = xy_split(Xy)
        self.fit(recursion=False, **Xy_train)
        Xy_test = self.predict(recursion=False, **Xy_test)
        return Xy_test

    def cv_grid_search(self, key2, result, cv_node):
        #print node, key2, result
        match_key3 = [key3 for key3 in result
            if re.search(self.key3, str(key3))]
        if len(match_key3) != 1:
            raise ValueError("None or more than one tertiary key found")
        # 1) Retrieve pairs of optimal (argument-name, value)
        key3 = match_key3[0]
        grid_cv = result[key3]
        mean_cv = np.mean(np.array(grid_cv), axis=0)
        mean_cv_opt = np.max(mean_cv) if self.arg_max else  np.min(mean_cv)
        idx_best = np.where(mean_cv == mean_cv_opt)
        idx_best = [idx[0] for idx in idx_best]
        grid = grid_cv[0]
        args_best = list()
        while len(idx_best):
            idx = idx_best[0]
            args_best.append((grid.axis_name, grid.axis_values[idx]))
            idx_best = idx_best[1:]
            grid = grid[0]
        # Retrieve one node that match intermediary key
        leaf = cv_node.get_node(regexp=key2, stop_first_match=True)
        # get path from current node
        path = leaf.get_path_from_node(cv_node)
        estimators = list()
        for node in path:
            if not hasattr(node, "estimator"):  # Strip off non estimator nodes
                continue
            estimator_args = copy.deepcopy(node.signature_args)
            for i in xrange(len(estimator_args)):
                arg_best = args_best[0]
                args_best = args_best[1:]
                #print node.get_key(), "Set", arg_best[0], "=", arg_best[1]
                estimator_args[arg_best[0]] = arg_best[1]
            new_estimator = copy.deepcopy(node.estimator)
            new_estimator.__dict__.update(estimator_args)
            #print "=>", new_estimator
            new_estimator_node = Estimator(new_estimator)
            new_estimator_node.signature_args = estimator_args
            # Parameters should no appear in intermediary key
            new_estimator_node.signature2_args_str = "*"
            estimators.append(new_estimator_node)
        # Build the sequential pipeline
        from epac.workflow.pipeline import Pipe
        pipeline = Pipe(*estimators)
        return pipeline