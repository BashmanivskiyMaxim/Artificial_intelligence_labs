import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from utilities import visualize_classifier

# Завантаження та підготовка даних з файлу 'data_imbalance.txt'
input_file = "data_imbalance.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

# Розділення даних на класи для подальшої візуалізації
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])

# Візуалізація вхідних даних на графіку
plt.figure()
plt.scatter(
    class_0[:, 0],
    class_0[:, 1],
    s=75,
    facecolor="black",
    edgecolors="black",
    linewidths=1,
    marker="x",
)

plt.scatter(
    class_1[:, 0],
    class_1[:, 1],
    s=75,
    facecolor="white",
    edgecolors="black",
    linewidths=1,
    marker="o",
)

plt.title("Вхідні дані")

# Розділення набору даних на тренувальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)
# Визначення параметрів класифікатора, з можливістю балансування класів
params = {"n_estimators": 100, "max_depth": 4, "random_state": 0}
if len(sys.argv) > 1:
    if sys.argv[1] == "balance":
        params = {
            "n_estimators": 100,
            "max_depth": 4,
            "random_state": 0,
            "class_weight": "balanced",
        }
    else:
        raise TypeError("Invalid input argument; should be 'balance'")

# Ініціалізація та тренування класифікатора
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)

# Візуалізація результатів класифікації на тренувальному та тестовому наборі даних
visualize_classifier(classifier, X_train, y_train, "Тренувальний набір даних")

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, "Тестовий набір даних")

class_names = ["Class-0", "Class-1"]
print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(
    classification_report(
        y_train, classifier.predict(X_train), target_names=class_names
    )
)
print("#" * 40 + "\n")

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#" * 40 + "\n")

plt.show()
