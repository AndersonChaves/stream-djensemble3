from sklearn.metrics import silhouette_samples
import numpy as np
from pyclustering.cluster.silhouette import silhouette
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.silhouette import silhouette

from pyclustering.samples.definitions import SIMPLE_SAMPLES
from pyclustering.utils import read_sample



def calculate_silhouette(data, labels, custom_distance):
    # Calculate silhouette score manually
    n_samples = len(data)
    silhouette_vals = np.zeros(n_samples)
    for i in range(n_samples):
        a = data[i]
        cluster_label = labels[i]
        a_dist = np.mean([custom_distance(a, data[j]) for j in range(n_samples) if labels[j] == cluster_label])
        b_dist = min([np.mean([custom_distance(a, data[j]) for j in range(n_samples) if labels[j] != cluster_label])])
        silhouette_vals[i] = (b_dist - a_dist) / max(a_dist, b_dist)

    silhouette_score = np.mean(silhouette_vals)
    print("Silhouette Score:", silhouette_score)

def calculate_silhouette_pyclustering(data, labels, custom_distance):
    # Calculate Silhouette score
    from pyclustering.utils.metric import distance_metric, type_metric
    metric = distance_metric(type_metric.USER_DEFINED, func=custom_distance)
    score = silhouette(data, labels, metric=metric).process().get_score()
    return sum(score)/len(score)


