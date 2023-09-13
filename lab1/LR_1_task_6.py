import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Завантаження даних з файлу
data = pd.read_csv(
    "lab1/data_multivar_nb.txt", header=None, names=["Feature1", "Feature2", "Target"]
)

# Розділення даних на ознаки і мітки класів
X = data[["Feature1", "Feature2"]]
y = data["Target"]

# Розділення даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Нормалізація даних для SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Навчання моделі SVM
svm_model = SVC(kernel="linear", C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Прогноз на тестовому наборі для SVM
svm_predictions = svm_model.predict(X_test_scaled)

# Навчання та оцінка наївного байєсівського класифікатора
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)


# Функція для обчислення показників якості класифікації
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1


# Розрахунок показників для SVM
svm_accuracy, svm_precision, svm_recall, svm_f1 = calculate_metrics(
    y_test, svm_predictions
)

# Розрахунок показників для наївного байєсівського класифікатора
nb_accuracy, nb_precision, nb_recall, nb_f1 = calculate_metrics(y_test, nb_predictions)

# Виведення результатів
print("Результати для моделі SVM:")
print("Accuracy:", svm_accuracy)
print("Precision:", svm_precision)
print("Recall:", svm_recall)
print("F1 Score:", svm_f1)

print("\nРезультати для моделі наївного байєсівського класифікатора:")
print("Accuracy:", nb_accuracy)
print("Precision:", nb_precision)
print("Recall:", nb_recall)
print("F1 Score:", nb_f1)
