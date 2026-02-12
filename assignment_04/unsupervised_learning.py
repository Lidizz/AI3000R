import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score

# Step 1: Data Generation
# a) We create 4 distinct clusters with some overlap to make it challenging
# X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# b) We create a more complex dataset with varying cluster densities and overlaps
# X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=[0.5, 1.0, 1.5, 2.0], random_state=42)

# c) Alternatively, we can create a dataset with more overlap and less distinct clusters
# X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.5, random_state=42)

# d) For an even more challenging dataset, we can increase the number of clusters and reduce the separation
X, y_true = make_blobs(n_samples=500, centers=8, cluster_std=0.6, random_state=42)

# 2. Visualize the raw data (No labels)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, color='gray', alpha=0.6, edgecolor='k')
plt.title("Raw Unlabeled Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print(f"Dataset shape: {X.shape}")

# Step 2: K-Means Implementation (The "Parametric" Approach)

k_values = [2, 3, 4, 5, 6]
silhouette_scores = []

for k in k_values:
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    score = silhouette_score(X, cluster_labels)
    silhouette_scores.append(score)
    print(f"Silhouette Score for k={k}: {score:.4f}")

best_k = k_values[np.argmax(silhouette_scores)]
print(f"Best number of clusters by Silhouette Score: {best_k}")

# Step 3: MeanShift Implementation (The "Non-Parametric" Approach)

# Estimate bandwidth automatically
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
print(f"\nEstimated bandwidth: {bandwidth:.4f}")

# Train MeanShift model
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_labels = meanshift.fit_predict(X)

# Find number of clusters discovered
n_clusters_meanshift = len(np.unique(meanshift_labels))
print(f"Number of clusters found by MeanShift: {n_clusters_meanshift}")

# Calculate Silhouette Score
meanshift_silhouette = silhouette_score(X, meanshift_labels)
print(f"MeanShift Silhouette Score: {meanshift_silhouette:.4f}")

# Visualize MeanShift clusters
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=meanshift_labels, s=50, cmap='viridis', alpha=0.6, edgecolor='k')
plt.scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1], 
            c='red', s=200, alpha=0.75, marker='X', edgecolor='black', linewidth=2)
plt.title(f"MeanShift Clustering (Found {n_clusters_meanshift} clusters)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# Step 4: Evaluation & Conclusion

print("\n" + "="*60)
print("FINAL COMPARISON")
print("="*60)

# Best K-Means result
best_kmeans_score = max(silhouette_scores)
print(f"\n1. K-Means Best Result:")
print(f"   - Optimal k: {best_k}")
print(f"   - Silhouette Score: {best_kmeans_score:.4f}")

# MeanShift result
print(f"\n2. MeanShift Result:")
print(f"   - Clusters found automatically: {n_clusters_meanshift}")
print(f"   - Silhouette Score: {meanshift_silhouette:.4f}")

# Comparison
print(f"\n3. Winner:")
if meanshift_silhouette > best_kmeans_score:
    print(f"   🏆 MeanShift performs better!")
    print(f"   (Silhouette: {meanshift_silhouette:.4f} vs {best_kmeans_score:.4f})")
elif meanshift_silhouette < best_kmeans_score:
    print(f"   🏆 K-Means performs better!")
    print(f"   (Silhouette: {best_kmeans_score:.4f} vs {meanshift_silhouette:.4f})")
if meanshift_silhouette == best_kmeans_score:
    print(f"   🏆 MeanShift & K-Means perform equally well!")
    print(f"   (Silhouette: {meanshift_silhouette:.4f} == {best_kmeans_score:.4f})")


# Visualize K-Means vs MeanShift side by side
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Best K-Means
kmeans_best = KMeans(n_clusters=best_k, n_init=10, random_state=42)
kmeans_best_labels = kmeans_best.fit_predict(X)

axes[0].scatter(X[:, 0], X[:, 1], c=kmeans_best_labels, s=50, cmap='viridis', alpha=0.6, edgecolor='k')
axes[0].scatter(kmeans_best.cluster_centers_[:, 0], kmeans_best.cluster_centers_[:, 1],
                c='red', s=200, alpha=0.75, marker='X', edgecolor='black', linewidth=2)
axes[0].set_title(f"K-Means (k={best_k})\nSilhouette: {best_kmeans_score:.4f}")
axes[0].set_xlabel("Feature 1")
axes[0].set_ylabel("Feature 2")

# MeanShift
axes[1].scatter(X[:, 0], X[:, 1], c=meanshift_labels, s=50, cmap='viridis', alpha=0.6, edgecolor='k')
axes[1].scatter(meanshift.cluster_centers_[:, 0], meanshift.cluster_centers_[:, 1],
                c='red', s=200, alpha=0.75, marker='X', edgecolor='black', linewidth=2)
axes[1].set_title(f"MeanShift ({n_clusters_meanshift} clusters)\nSilhouette: {meanshift_silhouette:.4f}")
axes[1].set_xlabel("Feature 1")
axes[1].set_ylabel("Feature 2")

plt.tight_layout()
plt.show()

print("\n" + "="*60)