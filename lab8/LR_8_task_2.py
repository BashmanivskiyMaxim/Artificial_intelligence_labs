import cv2
import numpy as np

# Зчитуємо вхідне зображення "Bashmanivskiy.jpg" і зберігаємо його в змінну 'img'.
img = cv2.imread("Bashmanivskiy.jpg")

# Визначаємо ядро для морфологічних операцій (діляції та ерозії).
kernel = np.ones((5, 5), np.uint8)

# Перетворюємо вхідне кольорове зображення в відтінки сірого та зберігаємо в 'imgGray'.
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Виконуємо розмиття Гауса на сірому зображенні та зберігаємо в 'imgBlur'.
imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 0)

# Застосовуємо детектор границь Canny на кольоровому зображенні та зберігаємо в 'imgCanny'.
imgCanny = cv2.Canny(img, 150, 200)

# Виконуємо морфологічну операцію діляції на зображенні 'imgCanny' та зберігаємо в 'imgDialation'.
imgDialation = cv2.dilate(imgCanny, kernel, iterations=1)

# Виконуємо морфологічну операцію ерозії на зображенні 'imgDialation' та зберігаємо в 'imgEroded'.
imgEroded = cv2.erode(imgDialation, kernel, iterations=1)

# Відображаємо оброблені зображення у вікнах з відповідними назвами.
cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.imshow("Canny Image", imgCanny)
cv2.imshow("Dialation Image", imgDialation)
cv2.imshow("Eroded Image", imgEroded)

# Чекаємо на натискання будь-якої клавіші перед закриттям вікон.
cv2.waitKey(0)
