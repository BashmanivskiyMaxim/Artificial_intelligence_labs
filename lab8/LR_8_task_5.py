import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Завантаження основного та зображення-шаблону
img = cv.imread("Bashmanivskiy.JPG", 0)
img2 = img.copy()
template = cv.imread("Bashmanivskiy_face.JPG", 0)
w, h = template.shape[::-1]

# Список методів порівняння
methods = [
    "cv.TM_CCOEFF",
    "cv.TM_CCOEFF_NORMED",
    "cv.TM_CCORR",
    "cv.TM_CCORR_NORMED",
    "cv.TM_SQDIFF",
    "cv.TM_SQDIFF_NORMED",
]

# Перебираємо методи порівняння
for meth in methods:
    img = img2.copy()
    method = eval(meth)

    # Застосовуємо метод шаблонного виявлення
    res = cv.matchTemplate(img, template, method)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

    # Визначаємо позицію результату в залежності від методу
    if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc

    bottom_right = (top_left[0] + w, top_left[1] + h)

    # Малюємо прямокутник навколо виявленого регіону
    cv.rectangle(img, top_left, bottom_right, 255, 2)

    # Відображаємо результати порівняння
    plt.subplot(121), plt.imshow(res, cmap="gray")
    plt.title("Matching Result"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img, cmap="gray")
    plt.title("Detected Point"), plt.xticks([]), plt.yticks([])
    plt.suptitle(meth)
    plt.show()
