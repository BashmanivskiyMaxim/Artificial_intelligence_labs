import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# Завантаження даних з файлу
data = np.genfromtxt("lab3/data_multivar_regr.txt", delimiter=",")

# Розділення даних на вхідні ознаки (X) та вихідну змінну (y)
X = data[:, :-1]
y = data[:, -1]

# Розбивка даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# Тренувальні дані
X_train, y_train = X[:num_training], y[:num_training]
# Тестові дані
X_test, y_test = X[num_training:], y[num_training:]


# Лінійний регресор
linear_regressor = linear_model.LinearRegression()
linear_regressor.fit(X_train, y_train)

# Прогноз для тестового набору даних
y_pred = linear_regressor.predict(X_test)

# Виведення метрик якості лінійної регресії
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_pred), 2))

# Поліноміальна регресія ступеня 10
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)

# Створення поліноміального регресора та навчання на тренувальних даних
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)

# Приклад точки даних для передбачення
datapoint = [[7.75, 6.35, 5.56]]

# Перетворення точки даних на поліном
poly_datapoint = polynomial.transform(datapoint)

# Прогноз з використанням лінійного та поліноміального регресорів
linear_prediction = linear_regressor.predict(datapoint)
poly_prediction = poly_linear_model.predict(poly_datapoint)

# Виведення результатів прогнозу
print("\nLinear regression prediction:", linear_prediction)
print("Polynomial regression prediction:", poly_prediction)
