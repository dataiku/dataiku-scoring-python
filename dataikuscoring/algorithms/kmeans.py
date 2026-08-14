import numpy as np

from .common import Clusterer


class KMeans(Clusterer):
    """Clusterer reproducing sklearn KMeans / MiniBatchKMeans.

    predict(X) assigns each row to the nearest cluster center (argmin squared-euclidean distance) in the
    preprocessed feature space and returns the cluster index; the model layer maps that index to the cluster
    name ("cluster_0", "cluster_1", ...). Both algorithms are fully defined by cluster_centers_, so one scorer
    covers both. Unlike Isolation Forest there is no anomaly score, so no decision_function is exposed.
    """

    def __init__(self, model_parameters):
        # k x n_features; row j is the center of cluster j
        self.cluster_centers = np.asarray(model_parameters["cluster_centers"], dtype=np.float64)
        # ||c||^2 per center, precomputed for the distance expansion below
        self._center_sq_norms = (self.cluster_centers ** 2).sum(axis=1)

    def predict(self, X):
        data = np.asarray(X, dtype=np.float64)
        # nearest center by squared-euclidean distance via the expansion ||x - c||^2 = ||x||^2 - 2 x.c + ||c||^2.
        # ||x||^2 is constant per row so it is dropped (it does not change the argmin); this keeps the cost
        # O(n*k) in memory and lets the Java engine reproduce the exact same formula for cross-engine parity.
        distances = -2.0 * data.dot(self.cluster_centers.T) + self._center_sq_norms
        return [int(i) for i in np.argmin(distances, axis=1)]

    def __repr__(self):
        return "KMeans(n_clusters={})".format(len(self.cluster_centers))


class MiniBatchKMeans(KMeans):
    """MiniBatchKMeans scores identically to KMeans (nearest center); the distinct class just preserves the
    model type through a serialize/reload round-trip."""

    def __repr__(self):
        return "MiniBatchKMeans(n_clusters={})".format(len(self.cluster_centers))
