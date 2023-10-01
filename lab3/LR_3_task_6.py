import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import PolynomialFeatures

# Згенеруємо випадкові дані
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)


#Створення поліноміальних ознак 10-го ступеня
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

# Створення моделі лінійної регресії
model_poly = LinearRegression()


# Створення моделі лінійної регресії
model = LinearRegression()

# Функція для обчислення MSE на навчальному та тестовому наборах
def plot_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')

    # Перевернути значення помилок MSE на позитивні
    train_scores = -train_scores
    test_scores = -test_scores

    # Обчислити середнє значення MSE для кожного розміру навчального набору
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)

    # Побудувати криві навчання
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Навчальний набір')
    plt.plot(train_sizes, test_mean, label='Тестовий набір')
    plt.xlabel('Розмір навчального набору')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Крива навчання')
    plt.legend()
    plt.grid(True)
    plt.show()

# Виклик функції для побудови кривих навчання
#plot_learning_curve(model, X, y)
plot_learning_curve(model_poly, X_poly, y)
