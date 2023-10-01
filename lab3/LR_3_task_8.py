import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

# Набір даних Iris
iris = datasets.load_iris()
X = iris.data  # Ознаки
y = iris.target  # Класи

# Кількість кластерів
num_clusters = 3  # Ірис має три класи

# Навчаняя моделі k-середніх на даних
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(X)

# Отримайте мітки кластерів для кожного об'єкта
labels = kmeans.labels_

# Визначення мапування між номерами кластерів і видами Iris
cluster_to_iris_mapping = {}
for cluster in range(num_clusters):
    cluster_samples = y[labels == cluster]
    most_common_iris = np.bincount(cluster_samples).argmax()
    cluster_to_iris_mapping[cluster] = most_common_iris

# Візуалізація результатів кластеризації
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow', s=50)
plt.xlabel('Довжина чашолистка')
plt.ylabel('Ширина чашолистка')
plt.title('Кластеризація набору даних Iris за допомогою k-середніх')

# Текстові мітки для кожного кластеру, позначені видами Iris
for i in range(num_clusters):
    cluster_center = kmeans.cluster_centers_[i]
    most_common_iris = cluster_to_iris_mapping[i]
    iris_name = iris.target_names[most_common_iris]
    plt.text(cluster_center[0], cluster_center[1], f'{iris_name}', fontsize=12, color='black', ha='center')

plt.show()
