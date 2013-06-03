# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:06:54 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
"""
import sys
import optparse
import time
import numpy as np

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import range_log2


def do_all(options):
    if options.k_max != "auto":
        k_values = range_log2(np.minimum(int(options.k_max),
                                         options.n_features), add_n=True)
    else:
        k_values = range_log2(options.n_features, add_n=True)
    C_values = [1, 10]
    #print options
    #sys.exit(0)
    if options.trace:
        from epac import conf
        conf.TRACE_TOPDOWN = True

    ## 1) Build dataset
    ## ================
    X, y = datasets.make_classification(n_samples=options.n_samples,
                                        n_features=options.n_features,
                                        n_informative=options.n_informative)

    ## 2) Build Workflow
    ## =================
    from epac import Perms, CV, CVBestSearchRefit, Pipe, Grid
    time_start = time.time()
    ## CV + Grid search of a pipeline with a nested grid search
    pipeline = CVBestSearchRefit(*[
                  Pipe(SelectKBest(k=k),
                      Grid(*[SVC(kernel="linear", C=C) for C in C_values]))
                  for k in k_values],
                  n_folds=options.n_folds_nested)

    #print pipeline.stats(group_by="class")
    wf = Perms(CV(pipeline, n_folds=options.n_folds),
             n_perms=options.n_perms, permute="y")
    print "Time ellapsed, tree construction:", time.time() - time_start

    ## 3) Run Workflow
    ## ===============
    time_fit_predict = time.time()
    wf.fit_predict(X=X, y=y)
    print "Time ellapsed, fit predict:",  time.time() - time_fit_predict
    time_reduce = time.time()

    ## 4) Reduce Workflow
    ## ==================
    wf.reduce()
    print "Time ellapsed, reduce:",   time.time() - time_reduce

if __name__ == "__main__":
    # Set default values to parameters
    n_samples = 100
    n_features = int(1E04)
    n_informative = 5
    n_perms = 100
    n_folds = 10
    n_folds_nested = 5
    k_max = "auto"
    # parse command line options
    parser = optparse.OptionParser()
    parser.add_option('-n', '--n_samples',
        help='(default %d)' % n_samples, default=n_samples, type="int")
    parser.add_option('-p', '--n_features',
        help='(default %d)' % n_features, default=n_features, type="int")
    parser.add_option('-i', '--n_informative',
        help='(default %d)' % n_informative, default=n_informative, type="int")
    parser.add_option('-m', '--n_perms',
        help='(default %d)' % n_perms, default=n_perms, type="int")
    parser.add_option('-f', '--n_folds',
        help='(default %d)' % n_folds, default=n_folds, type="int")
    parser.add_option('-g', '--n_folds_nested',
        help='(default %d)' % n_folds_nested, default=n_folds_nested, type="int")
    parser.add_option('-k', '--k_max',
        help='"auto": 1, 2, 4, ... n_features values. "fixed": 1, 2, 4, ..., k_max (default %s)' % k_max, default=k_max, type="string")
    parser.add_option('-t', '--trace',
        help='Trace execution (default %s)' % False, action='store_true', default=False)

    #argv = []
    #options, args = parser.parse_args(argv)
    options, args = parser.parse_args(sys.argv)
    do_all(options)
##python -m cProfile examples/large_toy.py >/tmp/large_toy_1000perm-10cv-5cv-1000p-100n.csv