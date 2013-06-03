# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:19:48 2013

@author: edouard.duchesnay@cea.fr

Test complex workflows made of combination of several primitives.
"""
import unittest

import string
import os.path
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import CVBestSearchRefit, Grid, Pipe, CV, Perms
from epac import SummaryStat, PvalPerms
from epac.sklearn_plugins import Permutations

class TestPermCV(unittest.TestCase):

    def test_perm_cv(self):
            X, y = datasets.make_classification(n_samples=20, n_features=5,
                                                n_informative=2)
            n_perms = 3
            n_folds = 2
            rnd = 0

            # = With EPAC
            wf = Perms(CV(SVC(kernel="linear"), n_folds=n_folds,
                                reducer=SummaryStat(keep=True)),
                                n_perms=n_perms, permute="y", 
                                random_state=rnd, reducer=None)
            r_epac = wf.fit_predict(X=X, y=y)

            # = With SKLEARN
            from sklearn.cross_validation import StratifiedKFold
            clf = SVC(kernel="linear")
            r_sklearn = [[None] * n_folds for i in xrange(n_perms)]
            perm_nb = 0
            for perm in Permutations(n=y.shape[0], n_perms=n_perms,
                                    random_state=rnd):
                y_p = y[perm]
                fold_nb = 0
                for idx_train, idx_test in StratifiedKFold(y=y_p,
                                                           n_folds=n_folds):
                    X_train = X[idx_train, :]
                    X_test = X[idx_test, :]
                    y_p_train = y_p[idx_train, :]
                    clf.fit(X_train, y_p_train)
                    r_sklearn[perm_nb][fold_nb] = clf.predict(X_test)
                    fold_nb += 1
                perm_nb += 1

            # Comparison
            comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
            self.assertTrue(comp, u'Diff Perm / CV: EPAC vs sklearn')

            # test reduce
            r_epac_reduce = wf.reduce().values()[0]['pred_te']
            comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
            self.assertTrue(comp, u'Diff Perm / CV: EPAC reduce')

class TestCVBestSearchRefit(unittest.TestCase):

    def test_perm_cv_grid_vs_sklearn(self):
        X, y = datasets.make_classification(n_samples=100, n_features=500,
                                            n_informative=5)
        n_perms = 3
        n_folds = 2
        n_folds_nested = 2
        random_state = 0
        k_values = [2, 3]
        C_values = [1, 10]
        # ===================
        # = With EPAC
        # ===================
        ## CV + Grid search of a pipeline with a nested grid search
        pipeline = CVBestSearchRefit(*[
                      Pipe(SelectKBest(k=k),
                          Grid(*[SVC(kernel="linear", C=C) for C in C_values]))
                      for k in k_values],
                      n_folds=n_folds_nested, random_state=random_state)
        wf = Perms(
                 CV(pipeline,
                       n_folds=n_folds,
                       reducer=SummaryStat(keep=True)),
                 n_perms=n_perms, permute="y",
                 reducer=PvalPerms(keep=True),
                 random_state=random_state)

        wf.fit_predict(X=X, y=y)
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
        from sklearn import grid_search

        clfs = {keys[0]: \
            Pipeline([('anova', SelectKBest(k=3)), ('svm', SVC(kernel="linear"))])}
        parameters = {'anova__k': k_values, 'svm__C': C_values}

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
        perms = Permutations(n=y.shape[0], n_perms=n_perms, random_state=random_state)
        for idx in perms:
            #idx = perms.__iter__().next()
            y_p = y[idx]
            cv = StratifiedKFold(y=y_p, n_folds=n_folds)
            fold_nb = 0
            for idx_train, idx_test in cv:
                #idx_train, idx_test  = cv.__iter__().next()
                X_train = X[idx_train, :]
                X_test = X[idx_test, :]
                y_p_train = y_p[idx_train, :]
                y_p_test = y_p[idx_test, :]
                for key in keys:
                    clf = clfs[key]
                    # Nested CV
                    cv_nested = StratifiedKFold(y=y_p_train, n_folds=n_folds_nested)
                    gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
                    gscv.fit(X_train, y_p_train)
                    R2[key]['pred_te'][perm_nb][fold_nb] = gscv.predict(X_test)
                    R2[key]['true_te'][perm_nb][fold_nb] = y_p_test
                    R2[key]['score_tr'][perm_nb][fold_nb] =\
                        gscv.score(X_train, y_p_train)
                    R2[key]['score_te'][perm_nb][fold_nb] =\
                        gscv.score(X_test, y_p_test)
                    fold_nb += 1
            for key in keys:
                # Average over folds
                R2[key]['mean_score_te'][perm_nb] = \
                    np.mean(np.asarray(R2[key]['score_te'][perm_nb]), axis=0)
                R2[key]['mean_score_tr'][perm_nb] = \
                    np.mean(np.asarray(R2[key]['score_tr'][perm_nb]), axis=0)
                    #np.mean(R2[key]['score_tr'][perm_nb])
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