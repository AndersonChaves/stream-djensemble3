import matplotlib.pyplot as plt

# Data
methods = ['Parcorr8', 'Parcorr10']
embedding = [0.0, 0.0]
num_clusters = [3.0, 3.0]
embedding_time = [0.0013, 1.6145]
clustering_time = [26.0854, 26.0176]
silhouette_score = [0.4883, 0.4760]

# Create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 10))

# Bar chart for Embedding and Number of Clusters
axes[0].bar(methods, embedding, label='Embedding', width=0.4, align='center')
axes[0].bar(methods, num_clusters, label='Number of Clusters', width=0.4, align='edge')
axes[0].set_title('Embedding and Number of Clusters')
axes[0].legend()

# Bar chart for Embedding Time, Clustering Time, and Silhouette Score
axes[1].bar(methods, embedding_time, label='Embedding Time', width=0.2, align='center')
axes[1].bar(methods, clustering_time, label='Clustering Time', width=0.2, align='edge')
axes[1].set_ylabel('Time (s)')
axes[1].legend()

axes[1].twinx()  # Create a secondary y-axis for Silhouette Score
axes[1].plot(methods, silhouette_score, marker='o', color='tab:orange', label='Silhouette Score')
axes[1].set_title('Embedding and Clustering Times vs. Silhouette Score')

# Show the plots
plt.tight_layout()
plt.show()