from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

import numpy as np

train_features = np.load('../data/train_SIFT_features.npy')[()]
all_SIFT = np.empty([1, 128])

for k, v in train_features.items():
    all_SIFT = np.concatenate([all_SIFT, v])
    
all_SIFT = np.delete(all_SIFT, 0, 0)

sil_score = []

for n_clusters in range(2, 21):

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(all_SIFT)

    silhouette_avg = silhouette_score(all_SIFT, cluster_labels)
    
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)
    
    sil_score.append(silhouette_avg)


plt.plot(range(2, 21), sil_score)
plt.title('Average Silhouette Score Scree Plot', fontsize=14)
ax = plt.axes()
ax.set_xticks(range(2, 21))
ax.set_xlabel('Number of Cluster (K)')
ax.set_ylabel('Average Silhouette Score')

kmeans = KMeans(n_clusters = 9, random_state = 10).fit(all_SIFT)

centers = kmeans.cluster_centers_

np.save('../data/cluster_centers.npy', centers)