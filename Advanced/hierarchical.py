import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# Load the Iris dataset
iris = load_iris()

# Create a DataFrame from the iris data
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Standardizing the data (important for clustering)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Perform hierarchical/agglomerative clustering with scikit-learn
model = AgglomerativeClustering(
    n_clusters=3, affinity='euclidean', linkage='ward')
model.fit(df_scaled)

# Visualize the clustering labels
df['Cluster'] = model.labels_

# Plot the clusters with a pairplot
sns.pairplot(df, hue='Cluster', palette='Set2', diag_kind='kde')
plt.show()

# For visualization of a dendrogram, we still need scipy's linkage method
linked = linkage(df_scaled, method='ward')

# Plot the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(linked,
           orientation='top',
           distance_sort='descending',
           show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram (scikit-learn)')
plt.xlabel('Sample index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()
