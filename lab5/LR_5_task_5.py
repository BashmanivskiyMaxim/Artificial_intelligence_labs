import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Генерація навчальних даних
min_val = -15
max_val = 15
num_points = 130
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Створення даних та міток
data = x.reshape(num_points, 1)
labels = y.reshape(num_points, 1)

# Відображення вхідних даних
plt.figure()
plt.scatter(data, labels)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Input data")

# Визначення багатошарової нейромережі з двома прихованими шарами;
# Перший прихований шар складається з 10 нейронів
# Другий прихований шар складається з 6 нейронів
# Вихідний шар складається з 1 нейрона
nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

# Встановлення алгоритму навчання на градієнтному спуску
nn.trainf = nl.train.train_gd

# Тренування нейромережі
error_progress = nn.train(data, labels, epochs=2000, show=100, goal=0.01)

# Виконання нейромережі на навчальних точках даних
output = nn.sim(data)
y_pred = output.reshape(num_points)

# Відображення помилки тренування
plt.figure()
plt.plot(error_progress)
plt.xlabel("Кількість епох")
plt.ylabel("Помилка")
plt.title("Прогрес помилки тренування")

# Відображення результату
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)

plt.figure()
plt.plot(x_dense, y_dense_pred, "-", x, y, ".", x, y_pred, "p")
plt.title("Фактичні дані порівняно з прогнозованими")

plt.show()
