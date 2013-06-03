#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:13:11 2013

@author: edouard.duchesnay@cea.fr
@author: benoit.da_mota@inria.fr
@author: jinpeng.li@cea.fr

Introduction
------------

The library Epac can create an Epac tree for machine learning algorithms.
This example shows how to compute Epac with n processes
"""

import os
import sys
import optparse
import shutil
import time
import numpy as np


from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import range_log2
from epac import StoreFs
from epac.export_multi_processes import  run_multi_processes

def do_all(options):
    '''
    + my_working_directory
      - epac_datasets.npz
      + epac_tree
    '''
    ## All the file paths should be ***RELATIVE*** path in the working directory
    # Training and test data
    datasets_file_relative_path = "./epac_datasets.npz"
    # root key for Epac tree
    tree_root_relative_path = "./epac_tree"
    random_state = 0

    ## 1) Create Working directory
    ## ===========================
    if os.path.isdir(options.working_dir_path):
        shutil.rmtree(options.working_dir_path)
    os.mkdir(options.working_dir_path)
    os.chdir(options.working_dir_path)

    ## 2) Build dataset
    ## ================
    X, y = datasets.make_classification(n_samples=options.n_samples,
                                        n_features=options.n_features,
                                        n_informative=options.n_informative)
    np.savez(datasets_file_relative_path, X=X, y=y)

    ## 3) Build Workflow
    ## =================
    from epac import Perms, CV, CVBestSearchRefit, Pipe, Grid
    from epac import SummaryStat, PvalPerms
    if options.k_max != "auto":
        k_values = range_log2(np.minimum(int(options.k_max),
                                         options.n_features), add_n=True)
    else:
        k_values = range_log2(options.n_features, add_n=True)
    C_values = [1, 10]
    time_start = time.time()
    ## CV + Grid search of a pipeline with a nested grid search
    pipeline = CVBestSearchRefit(*[
                  Pipe(SelectKBest(k=k),
                      Grid(*[SVC(kernel="linear", C=C) for C in C_values]))
                  for k in k_values],
                  n_folds=options.n_folds_nested, random_state=random_state)

    wf = Perms(CV(pipeline, n_folds=options.n_folds),
             n_perms=options.n_perms, permute="y", random_state=random_state)
    print "Time ellapsed, tree construction:", time.time() - time_start
    time_save = time.time()
    ## 4) Save on disk
    ## ===============
    store = StoreFs(dirpath=tree_root_relative_path)
    wf.save_tree(store=store)
    print "Time ellapsed, saving on disk:",  time.time() - time_save
    ## 5) Run
    ## ======
    time_fit_predict = time.time()
#    wf.fit_predict(X=X, y=y)
    run_multi_processes(
    in_datasets_file_relative_path=datasets_file_relative_path,
    in_working_directory=options.working_dir_path,
    in_tree_root=wf,
    in_num_cores=options.n_cores,
    in_is_wait=True
    )
    print "Time ellapsed, fit predict:",  time.time() - time_fit_predict
#    wf_key = wf.get_key()

    time_reduce = time.time()
    ## 6) Load Epac tree & Reduce
    ## ==========================
    wf = store.load()
    print wf.reduce()
    print "Time ellapsed, reduce:",   time.time() - time_reduce


if __name__ == "__main__":
    # Set default values to parameters
    n_samples = 100
    n_features = int(1E03)
    n_informative = 5
    n_perms = 10
    n_folds = 10
    n_folds_nested = 5
    k_max = "auto"
    n_cores = 2
    working_dir_path = "/tmp/my_working_directory"

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
    parser.add_option('-c', '--n_cores',
        help='(default %d)' % n_cores, default=n_cores, type="int")
    parser.add_option('-w', '--working_dir_path',
        help='(default %s)' % working_dir_path, default=working_dir_path)
    #argv = ['examples/large_toy.py']
    #options, args = parser.parse_args(argv)
    options, args = parser.parse_args(sys.argv)
    do_all(options)