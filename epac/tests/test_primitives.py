# -*- coding: utf-8 -*-
"""
Created on Sun May 19 19:29:17 2013

@author: edouard.duchesnay@cea.fr

Test simple EPAC primitives.
"""

import unittest
import numpy as np
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.lda import LDA
from sklearn.feature_selection import SelectKBest
import sklearn.pipeline
from epac import Pipe, Methods, CV, Perms
from epac import SummaryStat
from epac.sklearn_plugins import Permutations


class TestPipeline(unittest.TestCase):

    def test_pipeline(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)

        # = With EPAC
        wf = Pipe(SelectKBest(k=2), SVC(kernel="linear"))
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        pipe = sklearn.pipeline.Pipeline([('anova', SelectKBest(k=2)),
                         ('svm', SVC(kernel="linear"))])
        r_sklearn = pipe.fit(X, y).predict(X)

        # = Comparison
        self.assertTrue(np.all(r_epac == r_sklearn),
                        u'Diff in Pipe: EPAC vs sklearn')
        # test reduce
        r_epac_reduce = wf.reduce().values()[0]['pred_te']
        self.assertTrue(np.all(r_epac_reduce == r_sklearn),
                        u'Diff in Pipe: EPAC reduce')


class TestCV(unittest.TestCase):

    def test_cv(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_folds = 2

        # = With EPAC
        wf = CV(SVC(kernel="linear"), n_folds=n_folds,
                reducer=SummaryStat(keep=True))
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        from sklearn.cross_validation import StratifiedKFold
        clf = SVC(kernel="linear")
        r_sklearn = list()
        for idx_train, idx_test in StratifiedKFold(y=y, n_folds=n_folds):
            #idx_train, idx_test  = cv.__iter__().next()
            X_train = X[idx_train, :]
            X_test = X[idx_test, :]
            y_train = y[idx_train, :]
            clf.fit(X_train, y_train)
            r_sklearn.append(clf.predict(X_test))

        # = Comparison
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff CV: EPAC vs sklearn')

        # test reduce
        r_epac_reduce = wf.reduce().values()[0]['pred_te']
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff CV: EPAC reduce')


class TestPerms(unittest.TestCase):

    def test_perm(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        n_perms = 2
        rnd = 0

        # = With EPAC
        wf = Perms(SVC(kernel="linear"), n_perms=n_perms, permute="y",
                          random_state=rnd, reducer=None)
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        clf = SVC(kernel="linear")
        r_sklearn = list()
        for perm in Permutations(n=y.shape[0], n_perms=n_perms,
                                 random_state=rnd):
            y_p = y[perm, :]
            clf.fit(X, y_p)
            r_sklearn.append(clf.predict(X))

        # = Comparison
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm: EPAC vs sklearn')

        # test reduce
        r_epac_reduce = wf.reduce().values()[0]['pred_te']
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm: EPAC reduce')


class TestCVGridSearchRefit(unittest.TestCase):

    def test_cvgridsearchrefit(self):
        X, y = datasets.make_classification(n_samples=12, n_features=10, n_informative=2)
        from epac import CVGridSearchRefit
        # CV + Grid search of a simple classifier
        wf = CVGridSearchRefit(*[SVC(C=C) for C in [1, 10]], n_folds=2)
        wf.fit_predict(X=X, y=y)
        wf.reduce()

class TestMethods(unittest.TestCase):

    def test_constructor_avoid_collision_level1(self):
        # Test that level 1 collisions are avoided
        pm = Methods(*[SVC(kernel="linear", C=C) for C in [1, 10]])
        leaves_key = [l.get_key() for l in pm.walk_leaves()]
        self.assertTrue(len(leaves_key) == len(set(leaves_key)),
                        u'Collision could not be avoided')

    def test_constructor_avoid_collision_level2(self):
        # Test that level 2 collisions are avoided
        pm = Methods(*[Pipe(SelectKBest(k=2), SVC(kernel="linear", C=C))\
                          for C in [1, 10]])
        leaves_key = [l.get_key() for l in pm.walk_leaves()]
        self.assertTrue(len(leaves_key) == len(set(leaves_key)),
                        u'Collision could not be avoided')

    def test_constructor_cannot_avoid_collision_level2(self):
        # This should raise an exception since collision cannot be avoided
        self.assertRaises(ValueError, Methods,
                         *[Pipe(SelectKBest(k=2), SVC(kernel="linear", C=C))\
                          for C in [1, 1]])

    def test_twomethods(self):
        X, y = datasets.make_classification(n_samples=20, n_features=5,
                                            n_informative=2)
        # = With EPAC
        wf = Methods(LDA(), SVC(kernel="linear"))
        r_epac = wf.fit_predict(X=X, y=y)

        # = With SKLEARN
        lda = LDA()
        svm = SVC(kernel="linear")
        lda.fit(X, y)
        svm.fit(X, y)
        r_sklearn = [lda.predict(X), svm.predict(X)]

        # Comparison
        comp = np.all(np.asarray(r_epac) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Methods')

        # test reduce
        r_epac_reduce = [wf.reduce().values()[0]['pred_te'], 
            wf.reduce().values()[1]['pred_te']]
        comp = np.all(np.asarray(r_epac_reduce) == np.asarray(r_sklearn))
        self.assertTrue(comp, u'Diff Perm / CV: EPAC reduce')

if __name__ == '__main__':
    unittest.main()