import numpy as np
import neurolab as nl
import numpy.random as rand

# Задаємо стандартне відхилення для генерації випадкових даних
skv = 0.03

# Задаємо центри кластерів
centr = np.array([[0.2, 0.3], [0.4, 0.4], [0.7, 0.3], [0.2, 0.5], [0.4, 0.5]])

# Генеруємо випадкові дані навколо центрів з використанням стандартного відхилення
rand_norm = skv * rand.randn(100, 5, 2)
inp = np.array([centr + r for r in rand_norm])
inp.shape = (100 * 5, 2)

# Перемішуємо вхідні дані
rand.shuffle(inp)

# Створюємо мережу з двома входами та чотирма нейронами
net = nl.net.newc([[0.0, 1.0], [0.0, 1.0]], 5)

# Тренуємо мережу за допомогою алгоритму CWTA (Conscience Winner Take All)
error = net.train(inp, epochs=200, show=20)

# Відображення результатів
import pylab as pl

pl.title("Classification Problem")
pl.subplot(211)
pl.plot(error)
pl.xlabel("Epoch number")
pl.ylabel("error (default MAE)")
w = net.layers[0].np["w"]

pl.subplot(212)
pl.plot(
    inp[:, 0], inp[:, 1], ".", centr[:, 0], centr[:, 1], "yv", w[:, 0], w[:, 1], "p"
)
pl.legend(["train samples", "real centers", "train centers"])
pl.show()
