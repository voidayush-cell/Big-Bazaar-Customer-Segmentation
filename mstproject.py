import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score

# 1. Load Data (replace with actual Big Bazaar dataset)
# 1. Load Data
data = pd.read_csv("/Users/ayushkhandelwal/Desktop/Python/bigbazaar_customer_segments.csv", on_bad_lines='skip')

# Preview dataset
print("Columns:", data.columns)
print(data.head())

# 2. Data Cleaning
data.drop_duplicates(inplace=True)
data.fillna(data.mean(numeric_only=True), inplace=True)

# Choose features (adjust to your dataset)
features = ["Annual Income", "Spending Score", "Visit Frequency"]
X = data[features]

# 3. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------ K-MEANS ------------------
print("\n--- K-Means Clustering ---")

# Elbow method
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method (K-Means)")
plt.show()

# Fit KMeans (choose k=4 as example)
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
data["KMeans_Cluster"] = y_kmeans

# Evaluation
print("Silhouette Score (KMeans):", silhouette_score(X_scaled, y_kmeans))
print("Davies-Bouldin Score (KMeans):", davies_bouldin_score(X_scaled, y_kmeans))

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y_kmeans, palette="Set2")
plt.title("Customer Segments (K-Means)")
plt.show()

# ------------------ DBSCAN ------------------
print("\n--- DBSCAN Clustering ---")

dbscan = DBSCAN(eps=0.8, min_samples=5)  # tune eps & min_samples
y_dbscan = dbscan.fit_predict(X_scaled)
data["DBSCAN_Cluster"] = y_dbscan

# Evaluation (ignore noise = -1)
labels = y_dbscan[y_dbscan != -1]
features_eval = X_scaled[y_dbscan != -1]

if len(set(labels)) > 1:
    print("Silhouette Score (DBSCAN):", silhouette_score(features_eval, labels))
    print("Davies-Bouldin Score (DBSCAN):", davies_bouldin_score(features_eval, labels))
else:
    print("DBSCAN found too few clusters to evaluate.")

# Visualization
plt.figure(figsize=(8,6))
sns.scatterplot(x=X_scaled[:,0], y=X_scaled[:,1], hue=y_dbscan, palette="Set1")
plt.title("Customer Segments (DBSCAN)")
plt.show()

# ------------------ Cluster Profiles ------------------
print("\nKMeans Cluster Profiles:")
print(data.groupby("KMeans_Cluster")[features].mean())

print("\nDBSCAN Cluster Profiles:")
print(data.groupby("DBSCAN_Cluster")[features].mean())

# Save segmented dataset
data.to_csv("segmented_customers_clusters.csv", index=False)



