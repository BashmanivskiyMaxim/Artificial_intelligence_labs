import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Завантаження даних з файлу "data_perceptron.txt"
text = np.loadtxt('data_perceptron.txt')

# Виділення вхідних даних та вихідних міток з файлу
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

# Створення нової фігури для візуалізації даних
plt.figure()
plt.scatter(data[:,0], data[:,1])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('Input data')

# Встановлення меж для відображення графіку
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# Визначення кількості виходів для перцептрона
num_output = labels.shape[1]

# Встановлення діапазонів значень на осях x та y для вхідних даних
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]

# Створення нового перцептрона з використанням бібліотеки Neurolab
perceptron = nl.net.newp([dim1, dim2], num_output)

# Тренування перцептрона та збереження результатів
error_progress = perceptron.train(data, labels, epochs=100, show=20, lr=0.03)

# Створення нової фігури для відображення графіку помилки тренування
plt.figure()
plt.plot(error_progress)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()

# Відображення фігур
plt.show()
