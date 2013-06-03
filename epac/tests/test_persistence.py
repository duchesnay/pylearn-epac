# -*- coding: utf-8 -*-
"""
Created on Sun May 19 20:32:20 2013

@author: edouard.duchesnay@cea.fr

Test persistence
"""

import unittest

import string
import os.path
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import Pipe, Methods, CV, Perms
from epac import SummaryStat, PvalPerms
from epac import StoreFs
from epac import CVBestSearchRefit, Grid
from epac.sklearn_plugins import Permutations


class TestWorkFlow(unittest.TestCase):

    def test_peristence_load_and_fit_predict(self):
        X, y = datasets.make_classification(n_samples=20, n_features=10,
                                        n_informative=2)
        n_folds = 2
        n_folds_nested = 3
        k_values = [1, 2]
        C_values = [1, 2]
        pipeline = CVBestSearchRefit(*[
                      Pipe(SelectKBest(k=k),
                          Grid(*[SVC(kernel="linear", C=C) for C in C_values]))
                      for k in k_values],
                      n_folds=n_folds_nested)

        tree_mem = CV(pipeline, n_folds=n_folds,
                      reducer=SummaryStat(keep=False))
        # Save Tree
        import tempfile
        store = StoreFs(dirpath=tempfile.mkdtemp(), clear=True)
        tree_mem.save_tree(store=store) # save
        tree_mem.fit_predict(X=X, y=y)
        res_mem = tree_mem.reduce().values()[0]
        # Reload Tree
        tree_fs_noresults = store.load()
        tree_fs_noresults.fit_predict(X=X, y=y)
        res_fs_noresults = tree_fs_noresults.reduce().values()[0]
        # Save with results
        tree_fs_noresults.save_tree(store=store) # save
        tree_fs_withresults = store.load()
        res_fs_withresults = tree_fs_withresults.reduce().values()[0]
        #
        # Compare
        comp = np.all([
            np.all(np.asarray(res_mem[k])==np.asarray(res_fs_noresults[k])) and 
            np.all(np.asarray(res_fs_noresults[k])==np.asarray(res_fs_withresults[k]))
            for k in res_mem])
        self.assertTrue(comp)

    def test_peristence_perm_cv_parmethods_pipe_vs_sklearn(self):
        X, y = datasets.make_classification(n_samples=20,
                                        n_features=5, n_informative=2)
        n_perms = 3
        rnd = 0
        n_folds = 2
        k_values = [2, 3]
        # ===================
        # = With EPAC
        # ===================
        anovas_svm = Methods(*[Pipe(SelectKBest(k=k), SVC(kernel="linear"))
            for k in k_values])
        wf = Perms(CV(anovas_svm, n_folds=n_folds,
                             reducer=SummaryStat(keep=True)),
            n_perms=n_perms, permute="y", random_state=rnd,
            reducer=PvalPerms(keep=True))
        # Save workflow
        # -------------
        import tempfile
        #store = StoreFs("/tmp/toto", clear=True)
        store = StoreFs(tempfile.mktemp())
        wf.save_tree(store=store)
        wf = store.load()
        wf.fit_predict(X=X, y=y)
        ## Save results
        wf.save_tree(store=store)
        wf = store.load()
        ### Reload tree, all you need to know is the key
##        wf = WF.load(store=store, key=key)
        ### Reduces results
        R1 = wf.reduce()
        rm = os.path.dirname(os.path.dirname(R1.keys()[0])) + "/"
        R1 = {string.replace(key, rm, ""): R1[key] for key in R1}
        keys = R1.keys()
        # ===================
        # = Without EPAC
        # ===================
        from sklearn.cross_validation import StratifiedKFold
        from epac.sklearn_plugins import Permutations
        from sklearn.pipeline import Pipeline
        keys = R1.keys()
        import re

        def key_to_k(key):
            find_k = re.compile(u'k=([0-9])')
            return int(find_k.findall(key)[0])
        # ANOVA SVM-C
        clfs = {key: Pipeline([('anova', SelectKBest(k=key_to_k(key))),
                           ('svm', SVC(kernel="linear"))]) for key in keys}
        R2 = dict()
        for key in keys:
            R2[key] = dict()
            R2[key]['pred_te'] = [[None] * n_folds for i in xrange(n_perms)]
            R2[key]['true_te'] = [[None] * n_folds for i in xrange(n_perms)]
            R2[key]['score_tr'] = [[None] * n_folds for i in xrange(n_perms)]
            R2[key]['score_te'] = [[None] * n_folds for i in xrange(n_perms)]
            R2[key]['mean_score_te'] = [None] * n_perms
            R2[key]['mean_score_tr'] = [None] * n_perms
        perm_nb = 0
        perms = Permutations(n=y.shape[0], n_perms=n_perms, random_state=rnd)
        for idx in perms:
            y_p = y[idx]
            cv = StratifiedKFold(y=y_p, n_folds=n_folds)
            fold_nb = 0
            for idx_train, idx_test in cv:
                X_train = X[idx_train, :]
                X_test = X[idx_test, :]
                y_p_train = y_p[idx_train, :]
                y_p_test = y_p[idx_test, :]
                for key in keys:
                    clf = clfs[key]
                    clf.fit(X_train, y_p_train)
                    R2[key]['pred_te'][perm_nb][fold_nb] = clf.predict(X_test)
                    R2[key]['true_te'][perm_nb][fold_nb] = y_p_test
                    R2[key]['score_tr'][perm_nb][fold_nb] =\
                        clf.score(X_train, y_p_train)
                    R2[key]['score_te'][perm_nb][fold_nb] =\
                        clf.score(X_test, y_p_test)
                fold_nb += 1
            for key in keys:
                # Average over folds
                R2[key]['mean_score_te'][perm_nb] = \
                    np.mean(np.asarray(R2[key]['score_te'][perm_nb]), axis=0)
                R2[key]['mean_score_tr'][perm_nb] = \
                    np.mean(np.asarray(R2[key]['score_tr'][perm_nb]), axis=0)
            perm_nb += 1
        # ===================
        # = Comparison
        # ===================
        comp = dict()
        for key in R1:
            r1 = R1[key]
            r2 = R2[key]
            comp[key] = {k: np.all(np.asarray(r1[k]) == np.asarray(r2[k])) \
                for k in set(r1.keys()).intersection(set(r2.keys()))}
        #return comp
        for key in comp:
            for subkey in comp[key]:
                self.assertTrue(comp[key][subkey],
                u'Diff for key: "%s" and attribute: "%s"' % (key, subkey))

if __name__ == '__main__':
    unittest.main()