import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings("ignore")


class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, estimator, **parameters):
        self.estimator = estimator.set_params(**parameters)
        self.cols = None

    def fit(self, x, y=None):
        x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.cols = x.columns
        return self

    def transform(self, x):
        x_imputed = self.apply_imputer(x)
        return pd.DataFrame(x_imputed,
                            columns=self.cols)

    def apply_imputer(self, x):
        imputer = IterativeImputer(estimator=self.estimator, random_state=0)
        return imputer.fit_transform(x)

    def score(self, x, y_true):
        return r2_score(y_true, self.transform(x))


def em_imputer(data, num_iters=3):
    x = data.copy()
    mask = x != x
    mask = np.array(mask)
    x = SimpleImputer(strategy='mean').fit_transform(x)
    gmm = GaussianMixture(init_params='kmeans', random_state=0)
    for it in range(num_iters):
        gmm.fit(x)
        for row in range(x.shape[0]):
            if mask[row].sum():
                inv_cov = np.linalg.inv(gmm.covariances_[0, ~ mask[row]][:, ~ mask[row]])
                delta = x[row, ~ mask[row]] - gmm.means_[0, ~ mask[row]]
                coef = gmm.covariances_[0, mask[row]][:, ~ mask[row]].dot(inv_cov)
                x[row, mask[row]] = gmm.means_[0, mask[row]] + coef.dot(delta)

    return pd.DataFrame(x)


class EM(BaseEstimator, TransformerMixin):

    def __init__(self, num_iters=3):
        self.num_iters = num_iters
        self.cols = None

    def fit(self, x, y=None):
        x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.cols = x.columns
        return self

    def transform(self, x):
        x_imputed = self.apply_imputer(x)
        return pd.DataFrame(x_imputed,
                            columns=self.cols)

    def apply_imputer(self, x):
        df_filled = em_imputer(x, self.num_iters)
        return df_filled

    def score(self, x, y_true):
        return r2_score(y_true, self.transform(x))


def kmeans_missing(dataset, n_clusters=5):
    mask = dataset != dataset
    x_hat = SimpleImputer(strategy='mean').fit_transform(dataset)
    cls = KMeans(n_clusters)
    labels = cls.fit_predict(x_hat)
    centroids = cls.cluster_centers_
    x_hat[mask] = centroids[labels][mask]
    return pd.DataFrame(x_hat)


class Kmeans(BaseEstimator, TransformerMixin):

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.cols = None

    def fit(self, x, y=None):
        x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.cols = x.columns
        return self

    def transform(self, x):
        x_imputed = self.apply_imputer(x)
        return pd.DataFrame(x_imputed,
                            columns=self.cols)

    def apply_imputer(self, x):
        df_filled = kmeans_missing(x, self.n_clusters)
        return df_filled

    def score(self, x, y_true):
        return r2_score(y_true, self.transform(x))


class KNN(BaseEstimator, TransformerMixin):

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.cols = None

    def fit(self, x, y=None):
        x = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x)
        self.cols = x.columns
        return self

    def transform(self, x):
        x_imputed = self.apply_imputer(x)
        return pd.DataFrame(x_imputed,
                            columns=self.cols)

    def apply_imputer(self, x):
        imputer = KNNImputer(n_neighbors=self.n_neighbors)
        imputer.fit(x)
        return pd.DataFrame(imputer.transform(x))

    def score(self, x, y_true):
        return r2_score(y_true, self.transform(x))
