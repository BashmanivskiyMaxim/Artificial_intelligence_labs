import cv2

# Завантажуємо класифікатор для виявлення обличчя (haarcascade_frontalface_default.xml).
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Завантажуємо вхідне зображення "Bashmanivskiy.jpg".
img = cv2.imread("Bashmanivskiy.jpg")

# Перетворюємо кольорове зображення в відтінки сірого (зменшуємо обсяг кольорової інформації).
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Використовуючи класифікатор, виявляємо обличчя на зображенні.
# Цей метод `detectMultiScale` шукає обличчя на сірому зображенні з параметрами (масштаб, мінімальні сусіди).
faces = faceCascade.detectMultiScale(imgGray, 1.1, 4)

# Для кожного виявленого обличчя обводимо його прямокутником на оригінальному зображенні.
for x, y, w, h in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Відображаємо зображення з виділеними обличчями.
cv2.imshow("Result", img)

# Чекаємо на натискання будь-якої клавіші перед закриттям вікна.
cv2.waitKey(0)
