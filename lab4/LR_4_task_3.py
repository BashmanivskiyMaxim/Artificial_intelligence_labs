import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

# Завантаження даних з вхідного файлу
input_file = "data_random_forests.txt"
data = np.loadtxt(input_file, delimiter=",")
X, y = data[:, :-1], data[:, -1]

# Розділення даних на класи
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розділення даних на навчальний і тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=5
)

# Визначення сітки параметрів для пошуку оптимальних параметрів
parameter_grid = [
    {"n_estimators": [100], "max_depth": [2, 4, 7, 12, 18]},
    {"max_depth": [4], "n_estimators": [25, 50, 100, 250]},
]

# Метрики, для яких будуть шукатися оптимальні параметри
metrics = ["precision_weighted", "recall_weighted"]

for metric in metrics:
    print("\n#### Searching for optimal parameters for", metric)

    # Створення класифікатора і пошук оптимальних параметрів
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0), parameter_grid, cv=5, scoring=metric
    )
    classifier.fit(X_train, y_train)

    # Виведення результатів пошуку оптимальних параметрів
    print("\nGrid scores for the parameter grid:")
    for i in range(0, len(classifier.cv_results_["params"])):
        print(
            classifier.cv_results_["params"][i],
            "-->",
            classifier.cv_results_["rank_test_score"][i],
        )
    print("\nBest parameters:", classifier.best_params_)

# Передбачення класів на тестовому наборі та виведення звіту про продуктивність
y_pred = classifier.predict(X_test)
print("\nPerformance report:\n")
print(classification_report(y_test, y_pred))
