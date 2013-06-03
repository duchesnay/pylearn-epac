# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:37:54 2013

@author: edouard.duchesnay@cea.fr
"""
from collections import Set
import copy
import warnings


class ResultSet(Set):
    """ResultSet is a dictionnary indexed by intermediry keys (key2). It contains
    Result.

    See also
    --------
    Result
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
            ret += res.key() + ":"
            ret += repr({k: res[k] for k in res if k != "key"}) + "\n"
        return ret

    def add(self, result):
        if result in self:
            msg = 'Key "%s" already in ResultSet, do not overwrite' % result.key()
            warnings.warn(msg)
            ## FIX ME
            print msg
        else:
            self.results.append(result)

    def values(self):
        return self.results

#    def add(self, key2, suffix, score=None, pred=None, true=None):
#        """
#        Parameters
#        ----------
#        key2 str
#            Secondary key
#
#        suffix str
#            Result.TRAIN or Result.TEST
#
#        score Any type
#            the score
#
#        """
#        if key2 in self:
#            res = self[key2]
#        else:
#            res = Result()
#            self[key2] = res
#        if score is not None:
#            res[Result.concat_key3(Result.SCORE, suffix)] = score
#        if pred is not None:
#            res[Result.concat_key3(Result.PRED, suffix)] = pred
#        if true is not None:
#            res[Result.concat_key3(Result.TRUE, suffix)] = true


class Result(dict):
    """Result is a record with a key, and a payload.

    Example
    -------
    r = Result(key='SVC(C=1)')
    r[Result.SCORE, Result.TRAIN] = 1.0
    Result
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

    def __setitem__(self, *args):
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

"""
run ../epac/results.py
r1 = Result('SVC(C=1)')
r1[Result.SCORE, Result.TRAIN] = 1.0
r2 = Result('SVC(C=10)')
r2[Result.SCORE, Result.TRAIN] = 1.0


results = ResultSet()
results.add(r1)
results.add(r2)
r1 = results['SVC(C=1)']
r1[Result.SCORE, Result.TEST] = 0.8
r2 = results['SVC(C=10)']
r2[Result.SCORE, Result.TEST] = 0.8
r2["toto"] = 0.8


result_list = [r1, r2]

"""