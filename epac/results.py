# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:37:54 2013

@author: edouard.duchesnay@cea.fr
"""
from collections import Set
import copy
import warnings


class ResultSet(Set):
    """ResultSet is a set of Result

    Example
    -------
    >>> from epac import Result, ResultSet
    >>> r1 = Result('SVC(C=1)', **dict(a=1, b=2))
    >>> r2 = Result('SVC(C=2)', **dict(a=1, b=2))
    >>> r3 = Result('SVC(C=3)', **dict(a=1, b=2))
    >>> r4 = Result('SVC(C=4)', **dict(a=1, b=2))
    >>> set1 = ResultSet(r1, r2)
    >>> set2 = ResultSet(r3, r4)
    >>> set3 = ResultSet(set1, set2)
    >>> set3.add(Result('SVC(C=5)', **dict(a=1, b=2)))
    """
    def __init__(self, *args):
        self.results = list()
        for arg in args:
            if isinstance(arg, ResultSet):
                for res in arg:
                    self.add(copy.copy(res))
            if isinstance(arg, Result):
                self.add(copy.copy(arg))

    def __contains__(self, result):
        for res in self.results:
            if res.key() == result.key():
                return True
        else:
            return False

    def __iter__(self):
        for res in self.results:
            yield res

    def __len__(self):
        return len(self.results)

    def __getitem__(self, key):
        for res in self.results:
            if res.key() == key:
                return res
        else:
            return None

    def __repr__(self):
        s = "["
        cpt = 1
        for res in self.results:
            if cpt == 1:
                s += repr(res)
            else:
                s += " " + repr(res)
            if cpt < len(self.results):
                s += ",\n"
            cpt += 1
        return s+"]"

    def add(self, result):
        if result in self:
            msg = 'Key "%s" already in ResultSet, do not overwrite' % \
                result.key()
            warnings.warn(msg)
            ## FIX ME
            print msg
        else:
            self.results.append(result)

    def values(self):
        return self.results

    def keys(self):
        return [r["key"] for r in self.results]


class Result(dict):
    """Result is a record with a "key", and a "payload".

    Example
    -------
    >>> from epac import Result
    >>> r1 = Result('SVC(C=1)', **dict(a=1, b=2))
    >>> r2 = Result(**dict(key='SVC(C=10)', a=1, b=2))
    >>> r1["foo"] = "bar"
    >>> r1[Result.SCORE, Result.TRAIN] = 1.0
    """
    TRAIN = "tr"
    TEST = "te"
    SCORE = "score"
    PRED = "pred"
    TRUE = "true"
    SEP = "_"
    PRINT_ORDER_REGEXP = ["key", ".*mean_score_te", ".*mean_score_tr",
                          ".*score_te", ".*score_tr"]

    def __init__(self, key=None, **kwargs):
        if key:
            super(Result, self).__setitem__("key", key)
        if dict:
            self.update(kwargs)

    def key(self):
        return self["key"]

    def payload(self):
        return {k: self[k] for k in self if k != "key"}

    def __setitem__(self, *args):
        """if """
        if isinstance(args[0], tuple):
            arg_name = Result.cat(*args[0])
        else:
            arg_name = args[0]
        arg_val = args[1]
        super(Result, self).__setitem__(arg_name, arg_val)

    def __repr__(self):
        ordered_keys = _order_from_regexp(items=self.keys(),
                           order_regexps=Result.PRINT_ORDER_REGEXP)
        s = "{"
        cpt = 1
        for k in ordered_keys:
            s += "'%s': %s" % (k, self[k])
            if cpt < len(ordered_keys):
                s += ", "
            cpt += 1
        return s + "}"

    @classmethod
    def cat(self, *parts):
        """Concatenate keys 3"""
        return self.SEP.join([str(part) for part in parts])

    @classmethod
    def stack(self, result_list):
        """Stack results list, test that all results have the same key.

         Example
        -------
        >>> from epac import Result
        >>> r1 = Result('SVC(C=1)', {'score_te': .5, 'score_tr': 1,})
        >>> r2 = Result('SVC(C=1)', {'score_te': .25, 'score_tr': .75,})
        >>> print Result.stack([r1, r2])
        {'score_tr': [1, 0.75], 'score_te': [0.5, 0.25], 'key': 'SVC(C=1)'}
        """
        key = result_list[0].key()
        payload = dict()
        for k in result_list[0]:
            payload[k] = list()
        for result in result_list:
            for k in payload:
                if k == "key" and result[k] != key:
                    raise KeyError("Stack results with differnet key")
                payload[k].append(result[k])
        payload.pop("key")
        return Result(key=key, payload=payload)

def _order_from_regexp(items, order_regexps):
    """Re-order list given regular expression listed by priorities

    Example:
    --------
    >>> _order_from_regexp(["aA", "aZ", "bb", "bY", "cZ", "aY"], [".*Y", ".*Z"])
    ['bY', 'aY', 'aZ', 'cZ', 'aA', 'bb']
    """
    import re
    ordered = list()
    for order_regexp in order_regexps:
        matched =  [item for item in items if re.search(order_regexp, item)]
        for item in matched:
            ordered.append(item)
            items.remove(item)
    ordered += items
    return ordered

"""
run /home/ed203246/git/pylearn-epac/epac/results.py

self = Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.16666666666666666, 'pred_te': np.array([1, 0, 1, 1, 1, 0]), 'key': 'CV(nb=1)/SVC(C=0.1)', 'mean_score_te': 0.25})
print self
#self = Result(**{'mean_score_te': 0.25, 'mean_score_tr': 1.0, 'key': 'SVC(C=5)'})

keys = self.keys()
ordered_keys = list()





import numpy as np
r1=Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.33333333333333331, 'mean_score_te': 0.33333333333333331, 'pred_te': np.array([1, 0, 1, 1, 0, 0]), 'key': 'CV(nb=1)/SVC(C=1)'})
r2=Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.33333333333333331, 'pred_te': np.array([1, 0, 1, 1, 0, 0]), 'key': 'CV(nb=1)/SVC(C=2)'})
r3=Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.33333333333333331, 'pred_te': np.array([1, 0, 1, 1, 0, 0]), 'key': 'CV(nb=1)/SVC(C=5)'})


Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.33333333333333331, 'pred_te': np.array([1, 0, 1, 1, 0, 0]), 'key': 'CV(nb=1)/SVC(C=10)'})
Result(**{'true_te': np.array([0, 1, 0, 1, 0, 1]), 'score_tr': 1.0, 'score_te': 0.33333333333333331, 'pred_te': np.array([1, 0, 1, 1, 0, 0]), 'key': 'CV(nb=1)/SVC(C=100)'})
"""