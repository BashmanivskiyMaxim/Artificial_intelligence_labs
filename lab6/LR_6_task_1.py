import numpy as np
from data import train_data, test_data
import numpy as np
from numpy.random import randn
import random


class RNN:
    # Класична рекурентна нейронна мережа

    def __init__(self, input_size, output_size, hidden_size=64):
        # Вес
        self.Whh = randn(hidden_size, hidden_size) / 1000
        self.Wxh = randn(hidden_size, input_size) / 1000
        self.Why = randn(output_size, hidden_size) / 1000

        # Зміщення
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        """
        Виконання фази прямого поширення нейронної мережі з
        використанням введених даних.
        Повернення підсумкової видачі та прихованого стану.
        - Вхідні дані у масиві однозначного вектора з формою (input_size, 1).
        """
        h = np.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = {0: h}

        # Виконання кожного кроку нейронної мережі RNN
        for i, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h

        # Підрахунок значення виводу
        y = self.Why @ h + self.by

        return y, h

    def backprop(self, d_y, learn_rate=2e-2):
        """
        Виконання фази зворотного розповсюдження RNN.
        - d_y (dL/dy) має форму (output_size, 1).
        - learn_rate є дійсним числом float.
        """
        n = len(self.last_inputs)

        # ОБчислення dL/dWhy і dL/dby.
        d_Why = d_y @ self.last_hs[n].T
        d_by = d_y

        # Ініціалізація dL/dWhh, dL/dWxh, і dL/dbh до нуля.
        d_Whh = np.zeros(self.Whh.shape)
        d_Wxh = np.zeros(self.Wxh.shape)
        d_bh = np.zeros(self.bh.shape)

        # Обчислення dL/dh для останнього h.
        d_h = self.Why.T @ d_y

        # Зворотне розповсюдження по часу.
        for t in reversed(range(n)):
            # Среднее значение: dL/dh * (1 - h^2)
            temp = (1 - self.last_hs[t + 1] ** 2) * d_h

            # dL/db = dL/dh * (1 - h^2)
            d_bh += temp

            # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
            d_Whh += temp @ self.last_hs[t].T

            # dL/dWxh = dL/dh * (1 - h^2) * x
            d_Wxh += temp @ self.last_inputs[t].T

            # Далее dL/dh = dL/dh * (1 - h^2) * Whh
            d_h = self.Whh @ temp

        # Відсікаємо, щоб попередити розрив градієнтів.
        for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
            np.clip(d, -1, 1, out=d)

        # Обновляємо ваги і зміщення з використанням градієнтного спуску.
        self.Whh -= learn_rate * d_Whh
        self.Wxh -= learn_rate * d_Wxh
        self.Why -= learn_rate * d_Why
        self.bh -= learn_rate * d_bh
        self.by -= learn_rate * d_by


def processData(data, backprop=True):
    """
    Повернення втрат RNN і точності для даних
    - дані подані як словник, що відображує текст як True або False.
    - backprop визначає, чи потрібно використовувати звороднє розподілення
    """
    items = list(data.items())
    random.shuffle(items)

    loss = 0
    num_correct = 0

    for x, y in items:
        inputs = createInputs(x)
        target = int(y)

        # Пряме розподілення
        out, _ = rnn.forward(inputs)
        probs = softmax(out)

        # Обчислення втрат / точності
        loss -= np.log(probs[target])
        num_correct += int(np.argmax(probs) == target)

        if backprop:
            # Создание dL/dy
            d_L_d_y = probs
            d_L_d_y[target] -= 1

            # Зворотне розподілення
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)


def createInputs(text):
    """
    Повертає масив унітарних векторів
        які представляють слова у введеному рядку тексту
        - текст є рядком string
        - Унітарний вектор має форму (vocab_size, 1)
    """
    inputs = []
    for w in text.split(" "):
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)

    return inputs


# Створити словник
vocab = list(set([w for text in train_data.keys() for w in text.split(" ")]))
vocab_size = len(vocab)

print("%d unique words found" % vocab_size)  # знайдено 18 унікальних слів

# Призначити індекс кожному слову
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for i, w in enumerate(vocab)}

print(word_to_idx["good"])  # 16 (це може змінитися)
print(idx_to_word[0])  # сумно (це може змінитися)


def softmax(xs):
    # Застосування функції Softmax для вхідного масиву
    return np.exp(xs) / sum(np.exp(xs))


# Ініціалізація нашої рекурентної нейронної мережі RNN
rnn = RNN(vocab_size, 2)


inputs = createInputs("i am very good")
out, h = rnn.forward(inputs)
probs = softmax(out)
print(probs)  # [[0.50000095], [0.49999905]]

# Цикл для кожного прикладу тренування
for x, y in train_data.items():
    inputs = createInputs(x)
    target = int(y)

    # Пряме разповсюдження
    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    # Створення dL/dy
    d_L_d_y = probs
    d_L_d_y[target] -= 1

    # Зворотнє разповсюдження
    rnn.backprop(d_L_d_y)

# Цикл тренування
for epoch in range(1000):
    train_loss, train_acc = processData(train_data)

    if epoch % 100 == 99:
        print("--- Epoch %d" % (epoch + 1))
        print("Train:\tLoss %.3f | Accuracy: %.3f" % (train_loss, train_acc))

        test_loss, test_acc = processData(test_data, backprop=False)
        print("Test:\tLoss %.3f | Accuracy: %.3f" % (test_loss, test_acc))
