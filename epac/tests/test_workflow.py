# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 19:19:48 2013

@author: edouard.duchesnay@cea.fr

Test complex workflows made of combination of several primitives.
"""
import unittest

#import string
#import os.path
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from epac import CVBestSearchRefit, Pipe, CV, Perms, Methods
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
            r_epac_reduce = [v['pred_te'] for v in wf.reduce().values()]
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

        # = With EPAC
        pipelines = Methods(*[Pipe(SelectKBest(k=k), Methods(*[SVC(C=C) for C in C_values])) for k in k_values])
        #print [n for n in pipelines.walk_leaves()]
        pipelines_cv = CVBestSearchRefit(pipelines,
                        sn_folds=n_folds_nested, random_state=random_state)
        wf = Perms(CV(pipelines_cv,n_folds=n_folds, reducer=SummaryStat(keep=True)),
                 n_perms=n_perms, permute="y",
                 reducer=PvalPerms(keep=True),
                 random_state=random_state)
        wf.fit_predict(X=X, y=y)
        r_epac = wf.reduce().values()[0]

        # = With SKLEARN
        from sklearn.cross_validation import StratifiedKFold
        from epac.sklearn_plugins import Permutations
        from sklearn.pipeline import Pipeline
        from sklearn import grid_search

        clf = Pipeline([('anova', SelectKBest(k=3)), ('svm', SVC(kernel="linear"))])
        parameters = {'anova__k': k_values, 'svm__C': C_values}

        r_sklearn = dict()
        r_sklearn['pred_te'] = [[None] * n_folds for i in xrange(n_perms)]
        r_sklearn['true_te'] = [[None] * n_folds for i in xrange(n_perms)]
        r_sklearn['score_tr'] = [[None] * n_folds for i in xrange(n_perms)]
        r_sklearn['score_te'] = [[None] * n_folds for i in xrange(n_perms)]
        r_sklearn['mean_score_te'] = [None] * n_perms
        r_sklearn['mean_score_tr'] = [None] * n_perms

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
                # Nested CV
                cv_nested = StratifiedKFold(y=y_p_train, n_folds=n_folds_nested)
                gscv = grid_search.GridSearchCV(clf, parameters, cv=cv_nested)
                gscv.fit(X_train, y_p_train)
                r_sklearn['pred_te'][perm_nb][fold_nb] = gscv.predict(X_test)
                r_sklearn['true_te'][perm_nb][fold_nb] = y_p_test
                r_sklearn['score_tr'][perm_nb][fold_nb] =\
                    gscv.score(X_train, y_p_train)
                r_sklearn['score_te'][perm_nb][fold_nb] =\
                    gscv.score(X_test, y_p_test)
                fold_nb += 1
            # Average over folds
            r_sklearn['mean_score_te'][perm_nb] = \
                np.mean(np.asarray(r_sklearn['score_te'][perm_nb]), axis=0)
            r_sklearn['mean_score_tr'][perm_nb] = \
                np.mean(np.asarray(r_sklearn['score_tr'][perm_nb]), axis=0)
                    #np.mean(R2[key]['score_tr'][perm_nb])
            perm_nb += 1

        # = Comparison
        shared_keys = set(r_epac.keys()).intersection(set(r_sklearn.keys()))
        comp = {k: np.all(np.asarray(r_epac[k]) == np.asarray(r_sklearn[k]))
                for k in shared_keys}
        #return comp
        for key in comp:
            self.assertTrue(comp[key], u'Diff for attribute: "%s"' % key)



if __name__ == '__main__':
    unittest.main()