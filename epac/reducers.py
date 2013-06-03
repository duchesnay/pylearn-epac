# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 16:12:26 2013

Reducers for EPAC
@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import re
from abc import abstractmethod
#from epac.configuration import conf
from epac.results import Result

## ======================================================================== ##
## == Reducers                                                           == ##
## ======================================================================== ##


class Reducer(object):
    """ Reducer abstract class, called within the reduce method to process
    up-stream data flow of Result.

    Inherited classes should implement reduce(result)."""
    @abstractmethod
    def reduce(self, result):
        """Reduce abstract method

        Parameters
        ----------

        result: (dict)
            A result
        """


class SummaryStat(Reducer):
    """Reducer that select sub-result(s) items according to select_regexp, and
    reduce the sub-result(s) using the statistics stat.
    
    select_regexp: srt
      A string to select items (defaults "score_te")

    keep: boolean
      Should other items be kept (False) into summarized results.
      (default False)

    Example:
from epac import SummaryStat
result = [{'key': 'LDA', 'score_te': 0.0}, {'key': 'LDA', 'score_te': 0.5}]


    >>> from epac import SummaryStat
    >>> result = Result('LDA', {'score_tr': [1, .8], 'score_te': [.9, .7]})
    >>> print SummaryStat(keep=True).reduce(result)
    {'score_te': [0.9, 0.7], 'mean_score_te': 0.80000000000000004, 'mean_score_tr': 0.90000000000000002, 'score_tr': [1, 0.8]}
    >>> print SummaryStat(keep=False).reduce(result)
    {'mean_score_te': 0.80000000000000004, 'mean_score_tr': 0.90000000000000002}
    """
    def __init__(self, select_regexp='score', stat="mean",
                 keep=False):
        self.select_regexp = select_regexp
        self.stat = stat
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            select_key3s = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
        else:
            select_key3s = result.keys()
        out = Result(key=result.key())
        for key3 in select_key3s:
            if self.stat == "mean":
                out[self.stat, key3] = np.mean(np.array(result[key3]), axis=0)
        if self.keep:
            out.update(result)
        return out


class PvalPerms(Reducer):
    """Reducer that select sub-result(s) according to select_regexp, and
    reduce the sub-result(s) using the statistics stat"""
    def __init__(self, select_regexp='mean.*' + Result.SCORE,
                 keep=False):
        self.select_regexp = select_regexp
        self.keep = keep

    def reduce(self, result):
        if self.select_regexp:
            select_keys = [key3 for key3 in result
                if re.search(self.select_regexp, str(key3))]
                #if re.search(self.select_regexp) != -1]
        else:
            select_keys = result.keys()
        out = Result(key=result.key())
        for key3 in select_keys:
            out[key3] = result[key3][0]
            count = np.sum(result[key3][1:] > result[key3][0])
            pval = float(count) / (len(result[key3]) - 1)
            #out[Result.concat_key3("count", str(key3))] = count
            out["pval", key3] = pval
        if self.keep:
            out.update(result)
        return out