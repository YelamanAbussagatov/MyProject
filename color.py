import cv2
import numpy as np

# Задание диапазонов цветов для распознавания
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_blue = np.array([110, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_green = np.array([50, 100, 100])
upper_green = np.array([70, 255, 255])

# Захват видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()

    # Преобразование кадра из BGR в HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Создание масок для каждого цвета
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # Применение масок к оригинальному кадру
    red_result = cv2.bitwise_and(frame, frame, mask=red_mask)
    blue_result = cv2.bitwise_and(frame, frame, mask=blue_mask)
    green_result = cv2.bitwise_and(frame, frame, mask=green_mask)

    # Отображение результатов
    cv2.imshow('Red Objects', red_result)
    cv2.imshow('Blue Objects', blue_result)
    cv2.imshow('Green Objects', green_result)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()