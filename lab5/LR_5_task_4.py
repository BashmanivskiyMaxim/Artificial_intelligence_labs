import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Завантаження вхідних даних
text = np.loadtxt("data_simple_nn.txt")

# Розділення даних на вхідні точки та мітки
data = text[:, 0:2]
labels = text[:, 2:]

# Візуалізація вхідних даних
plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("Input data")

# Мінімальні та максимальні значення для кожної змінної
dim1_min, dim1_max = data[:, 0].min(), data[:, 0].max()
dim2_min, dim2_max = data[:, 1].min(), data[:, 1].max()

# Визначення кількості нейронів у вихідному шарі
num_output = labels.shape[1]

# Визначення одношарової нейромережі
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
nn = nl.net.newp([dim1, dim2], num_output)

# Тренування нейромережі
error_progress = nn.train(data, labels, epochs=100, show=20, lr=0.03)

# Відображення ходу тренування
plt.figure()
plt.plot(error_progress)
plt.xlabel("Number of epochs")
plt.ylabel("Training error")
plt.title("Training error progress")
plt.grid()

plt.show()

# Виконання класифікації на тестових точках
print("\nTest results:")
data_test = [[0.4, 4.3], [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, "-->", nn.sim([item])[0])
