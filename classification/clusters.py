import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

file_path = 'Семинар 9.xlsx'
data = pd.read_excel(file_path, sheet_name=1)
X = data[['x1', 'x2']].values

np.random.seed(5)

silhouette_scores = []
clusters_range = [2, 3, 4, 5]

for n_clusters in clusters_range:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=60)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                color='red', marker='X', s=100, label='Центроиды')
    plt.title(f'K-means, кластеров {n_clusters} ')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.show()

plt.figure(figsize=(8, 6))
plt.plot(clusters_range, silhouette_scores, marker='o')
plt.xlabel('Число кластеров')
plt.ylabel('Cилуэт')
plt.grid(True)
plt.show()

optimal_clusters = clusters_range[np.argmax(silhouette_scores)]
print(f'Оптимальное количество кластеров: {optimal_clusters}')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Параметры для DBSCAN
eps = 0.3
min_samples = 5

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_scaled)

X_df = pd.DataFrame(X, columns=['x1', 'x2'])
X_df['cluster'] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x1', y='x2', hue='cluster', palette='viridis', data=X_df, s=60)
plt.title(f'Метод DBSCAN (eps={eps}, min_точек={min_samples})')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(title='Кластер', loc='best')
plt.show()
