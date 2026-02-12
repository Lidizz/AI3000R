# Assignment 4: Unsupervised Learning

## Objective

Cluster a complex dataset using a centroid-based algorithm (K-Means) and a density-based algorithm (MeanShift), then evaluate which performs better using the Silhouette Coefficient.

---

## Step 1: The Prepared Dataset

Copy and run this code block first. This creates your environment and dataset.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 1. Generate Prepared Data (Fixed Random State for consistency)
# We create 4 distinct clusters with some overlap to make it challenging
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=42)

# 2. Visualize the raw data (No labels)
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], s=50, color='gray', alpha=0.6, edgecolor='k')
plt.title("Raw Unlabeled Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

print(f"Dataset shape: {X.shape}")
```

---

## Step 2: K-Means Implementation (The "Parametric" Approach)

### Task

1. K-Means requires you to specify the number of clusters ($k$) beforehand. However, we don't know the optimal $k$ yet.
2. Run K-Means for $k = [2, 3, 4, 5, 6]$.
3. For each $k$, calculate the Silhouette Score.
4. Print the scores and determine which $k$ provides the best separation.

**HINT:** A higher Silhouette Score (closer to +1) indicates better defined clusters.

---

## Step 3: MeanShift Implementation (The "Non-Parametric" Approach)

### Task

1. MeanShift does not require you to specify the number of clusters; it finds them automatically based on bandwidth (radius).
2. Train a MeanShift model on the same `X` data. Let it estimate the bandwidth automatically (or use `estimate_bandwidth` from sklearn).
3. Calculate the Silhouette Score for the resulting clusters.
4. Plot the final clusters.

---

## Step 4: Evaluation & Conclusion

### Compare the results

1. Which $k$ had the highest Silhouette Score in K-Means?
2. How many clusters did MeanShift find automatically?
3. Compare the MeanShift Silhouette Score to the best K-Means Silhouette Score. Which algorithm performed better for this specific data distribution?

---

## Notes

- **Silhouette Score Range:** -1 to +1
  - Values close to +1 indicate well-separated clusters
  - Values close to 0 indicate overlapping clusters
  - Negative values indicate misclassified points

- **K-Means** is a centroid-based algorithm that requires specifying the number of clusters
- **MeanShift** is a density-based algorithm that automatically discovers the number of clusters