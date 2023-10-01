import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# Дані з файлу data_clustering.txt
data = np.loadtxt("lab3/data_clustering.txt", delimiter=",")
X = data  # Ознаки

# Оцінюємо оптимальну ширину для Mean Shift
bandwidth = estimate_bandwidth(X, quantile=0.1)

# Застосовуємо алгоритм Mean Shift з використанням оптимальної ширини
ms = MeanShift(bandwidth=bandwidth)
ms.fit(X)

# Отримаємо мітки кластерів
labels = ms.labels_

# Отримаємо координати центрів кластерів
cluster_centers = ms.cluster_centers_

# Кількість кластерів знайдених алгоритмом Mean Shift
n_clusters = len(np.unique(labels))

# Виведкмо кількість знайдених кластерів
print(f"Кількість знайдених кластерів: {n_clusters}")

# Візуалізуємо результати кластеризації
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='black', marker='x', s=100)
plt.xlabel('Ознака 1')
plt.ylabel('Ознака 2')
plt.title('Кластеризація набору даних за допомогою Mean Shift')
plt.show()
