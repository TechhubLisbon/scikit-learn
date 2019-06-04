from joblib import Parallel, delayed
from time import time
import numpy as np
import pandas as pd
from sklearn.ensemble.iforest import IsolationForest, _average_path_length


class ParallelPredIsolationForest(IsolationForest):
    def __init__(self, n_estimators=100, max_samples="auto",
                 contamination="auto", max_features=1., bootstrap=False,
                 n_jobs=None, behaviour='deprecated', random_state=None,
                 verbose=0, warm_start=False):
        super().__init__(n_estimators=n_estimators, max_samples=max_samples,
                         contamination=contamination,
                         max_features=max_features, bootstrap=bootstrap,
                         n_jobs=n_jobs, behaviour=behaviour,
                         random_state=random_state, verbose=verbose,
                         warm_start=warm_start)

    def _compute_score_samples(self, X, subsample_features):
        """Compute the score of each samples in X going through the extra
        trees.
        Parameters
        ----------
        X : array-like or sparse matrix
        subsample_features : bool,
            whether features should be subsampled
        Returns
        -------
        ndarray
            The anomaly scores for each sample
        """

        def get_score(_X, _tree, _features, _subsample_features):
            X_subset = _X[:, _features] if _subsample_features else _X

            leaves_index = _tree.apply(X_subset)
            node_indicator = _tree.decision_path(X_subset)
            n_samples_leaf = _tree.tree_.n_node_samples[leaves_index]

            return np.ravel(node_indicator.sum(axis=1)) + _average_path_length(
                n_samples_leaf) - 1.0

        n_samples = X.shape[0]

        depths = np.zeros(n_samples, order="f")

        par_exec = Parallel(n_jobs=self.n_jobs, prefer="threads",
                            require='sharedmem')
        par_results = par_exec(
            delayed(get_score)(_X=X, _tree=tree, _features=features,
                               _subsample_features=subsample_features)
            for tree, features in zip(self.estimators_,
                                      self.estimators_features_))

        for result in par_results:
            depths += result

        scores = 2 ** (-depths / (len(self.estimators_)
                                  * _average_path_length([self.max_samples_])))

        return scores


def get_data(n_samples_train, n_samples_test, n_features, contamination=0.1,
             seed=19900603):
    """ Function based on code from: https://scikit-learn.org/stable/
    auto_examples/ensemble/plot_isolation_forest.html#sphx-glr-auto-
    examples-ensemble-plot-isolation-forest-py
    """
    rng = np.random.RandomState(seed)

    X = 0.3 * rng.randn(n_samples_train, n_features)
    X_train = np.r_[X + 2, X - 2]

    X = 0.3 * rng.randn(n_samples_test, n_features)
    X_test = np.r_[X + 2, X - 2]

    n_outliers = int(np.floor(contamination * n_samples_test))
    X_outliers = rng.uniform(low=-4, high=4, size=(n_outliers, n_features))

    outlier_idx = rng.choice(np.arange(0, n_samples_test), n_outliers,
                             replace=False)
    X_test[outlier_idx, :] = X_outliers

    return X_train, X_test


def test_n_rows(n_samples_list, n_jobs_list, n_features=30, repetitions=5,
                n_trees=100):
    assert isinstance(n_samples_list, list)
    assert isinstance(n_jobs_list, list)
    assert isinstance(n_trees, int)
    assert isinstance(n_features, int)
    assert isinstance(repetitions, int)

    results_sklearn_iforest = np.zeros((len(n_samples_list),
                                        len(n_jobs_list)))
    results_parallel_iforest = np.zeros((len(n_samples_list),
                                         len(n_jobs_list)))

    # First, let's evaluate current implementation
    for i, n_samples in enumerate(n_samples_list):
        X_train, X_test = get_data(n_samples_train=n_samples,
                                   n_samples_test=n_samples,
                                   n_features=n_features)
        rng = np.random.RandomState(19900603)
        for j, n_jobs in enumerate(n_jobs_list):
            for _ in range(repetitions):
                # Test current Isolation Forest
                iforest = IsolationForest(n_estimators=n_trees, n_jobs=n_jobs,
                                          random_state=rng)
                iforest.fit(X_train)
                start = time()
                _ = iforest.predict(X_test)
                results_sklearn_iforest[i, j] += time() - start

                # Test PR Isolation Forest
                iforest = ParallelPredIsolationForest(
                    n_estimators=n_trees, n_jobs=n_jobs, random_state=rng)
                iforest.fit(X_train)
                start = time()
                _ = iforest.predict(X_test)
                results_parallel_iforest[i, j] += time() - start

            results_sklearn_iforest[i, j] /= repetitions
            results_parallel_iforest[i, j] /= repetitions

    speed_up = results_sklearn_iforest / results_parallel_iforest

    results_sklearn_iforest = pd.DataFrame(
        results_sklearn_iforest, index=n_samples_list, columns=n_jobs_list)
    results_parallel_iforest = pd.DataFrame(
        results_parallel_iforest, index=n_samples_list, columns=n_jobs_list)
    speed_up = pd.DataFrame(
        speed_up, index=n_samples_list, columns=n_jobs_list)

    return results_sklearn_iforest, results_parallel_iforest, speed_up


def test_n_trees(n_trees_list, n_jobs_list, n_features=30, repetitions=5,
                 n_samples=100000):
    assert isinstance(n_trees_list, list)
    assert isinstance(n_jobs_list, list)
    assert isinstance(n_samples, int)
    assert isinstance(n_features, int)
    assert isinstance(repetitions, int)

    results_sklearn_iforest = np.zeros((len(n_trees_list),
                                        len(n_jobs_list)))
    results_parallel_iforest = np.zeros((len(n_trees_list),
                                         len(n_jobs_list)))

    # First, let's evaluate current implementation
    for i, n_trees in enumerate(n_trees_list):
        X_train, X_test = get_data(n_samples_train=n_samples,
                                   n_samples_test=n_samples,
                                   n_features=n_features)
        rng = np.random.RandomState(19900603)
        for j, n_jobs in enumerate(n_jobs_list):
            for _ in range(repetitions):
                # Test current Isolation Forest
                iforest = IsolationForest(n_estimators=n_trees, n_jobs=n_jobs,
                                          random_state=rng)
                iforest.fit(X_train)
                start = time()
                _ = iforest.predict(X_test)
                results_sklearn_iforest[i, j] += time() - start

                # Test PR Isolation Forest
                iforest = ParallelPredIsolationForest(
                    n_estimators=n_trees, n_jobs=n_jobs, random_state=rng)
                iforest.fit(X_train)
                start = time()
                _ = iforest.predict(X_test)
                results_parallel_iforest[i, j] += time() - start

            results_sklearn_iforest[i, j] /= repetitions
            results_parallel_iforest[i, j] /= repetitions

    speed_up = results_sklearn_iforest / results_parallel_iforest

    results_sklearn_iforest = pd.DataFrame(
        results_sklearn_iforest, index=n_trees_list, columns=n_jobs_list)
    results_parallel_iforest = pd.DataFrame(
        results_parallel_iforest, index=n_trees_list, columns=n_jobs_list)
    speed_up = pd.DataFrame(
        speed_up, index=n_trees_list, columns=n_jobs_list)

    return results_sklearn_iforest, results_parallel_iforest, speed_up
