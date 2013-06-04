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
    >>> result_set = ResultSet()
    >>> result_set.add(Result('SVC(C=1)',  payload=dict(score=.33)))
    >>> result_set.add(Result('SVC(C=10)', payload=dict(score=.55)))
    """

    def __init__(self, results_list=None):
        self.results = list()
        if results_list:
            for results in results_list:
                for result in results:
                    self.add(copy.copy(result))

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
        ret = ""
        for res in self.results:
            ret += repr(res) + "\n"
        return ret

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
    >>> r = Result('SVC(C=1)')
    >>> r["foo"] = "bar"
    >>> r[Result.SCORE, Result.TRAIN] = 1.0
    """
    TRAIN = "tr"
    TEST = "te"
    SCORE = "score"
    PRED = "pred"
    TRUE = "true"
    SEP = "_"

    def __init__(self, key, payload={}):
        #self["key"] = key
        super(Result, self).__setitem__("key", key)
        self.update(payload)

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
