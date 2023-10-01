import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Згенеруємо випадкові дані
np.random.seed(0)
m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)

# Виведемо дані на графіку
plt.scatter(X, y, label='Дані')
plt.xlabel('X')
plt.ylabel('y')

# Побудова лінійної регресії
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_lin = lin_reg.predict(X)

# Побудова поліноміальної регресії ступеня 2
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)
y_poly = lin_reg_poly.predict(X_poly)

# Виведення на графік
plt.plot(X, y_lin, label='Лінійна регресія', color='red')
plt.plot(X, y_poly, label='Поліноміальна регресія', color='green')
plt.legend()

# Оцінка якості моделей
mse_lin = mean_squared_error(y, y_lin)
r2_lin = r2_score(y, y_lin)
mse_poly = mean_squared_error(y, y_poly)
r2_poly = r2_score(y, y_poly)

coefficients = lin_reg_poly.coef_
intercept = lin_reg_poly.intercept_
print(coefficients)
print(intercept)



print("Лінійна регресія:")
print(f"Mean Squared Error (MSE): {mse_lin}")
print(f"R2 Score (коефіцієнт детермінації): {r2_lin}")

print("\nПоліноміальна регресія:")
print(f"Mean Squared Error (MSE): {mse_poly}")
print(f"R2 Score (коефіцієнт детермінації): {r2_poly}")

plt.show()
