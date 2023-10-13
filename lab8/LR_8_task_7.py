import cv2
import numpy as np
import random


def process_image(image_path):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 1
    )

    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(
        cont_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    area_groups = {
        "group_1": [],
        "group_2": [],
        "group_3": [],
        "group_4": [],
        "group_5": [],
        "group_6": [],
        "group_7": [],
        "group_8": [],
    }

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 2000 or area > 12000:
            continue

        if len(cnt) < 5:
            continue

        if 2000 < area < 3000:
            color = (255, 0, 0)
            area_groups["group_1"].append(area)
        elif 3000 < area < 4000:
            color = (34, 139, 34)
            area_groups["group_2"].append(area)
        elif 4000 < area < 5000:
            color = (0, 0, 255)
            area_groups["group_3"].append(area)
        elif 5000 < area < 6000:
            color = (255, 255, 0)
            area_groups["group_4"].append(area)
        elif 6000 < area < 7000:
            color = (0, 255, 255)
            area_groups["group_5"].append(area)
        elif 7000 < area < 8000:
            color = (255, 0, 255)
            area_groups["group_6"].append(area)
        elif 8000 < area < 9000:
            color = (128, 0, 0)
            area_groups["group_7"].append(area)
        else:
            color = (0, 128, 0)
            area_groups["group_8"].append(area)
        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(frame, ellipse, color, 2)

    cv2.imshow("coins bin ", cont_img)
    cv2.imshow("Morphological Closing", closing)
    cv2.imshow("Adaptive Thresholding", thresh)
    cv2.imshow("Contours", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "coins_3.png"
    process_image(image_path)
