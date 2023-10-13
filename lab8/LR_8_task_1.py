import cv2

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)
while True:
    success, img = cap.read()
    cv2.imshow("Result", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# LOAD AN IMAGE USING 'IMREAD'
img = cv2.imread("Bashmanivskiy.jpg")
# DISPLAY
cv2.imshow("Bashmanivskiy ",img)
cv2.waitKey(0)
