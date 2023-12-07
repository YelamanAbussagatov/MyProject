import cv2
import numpy as np
# Функция для распознавания формы объектов
def detect_shapes(c):
    # Периметр контура
    perimeter = cv2.arcLength(c, True)
    # Аппроксимируем контур
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

    # Определение формы
    if len(approx) == 3:
        return "Triangle"
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h
        if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
            return "Square"
        else:
            return "Rectangle"
    else:
           return "Circle"
# Открытие видеопотока с веб-камеры (0 - индекс веб-камеры)
cap = cv2.VideoCapture(0)

while True:
    # Захват кадра с веб-камеры
    ret, frame = cap.read()

    if not ret:
        break

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применение порогового значения для выделения контуров
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Поиск контуров
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Игнорируем маленькие контуры
        if cv2.contourArea(contour) < 100:
            continue
        # Получение координат центра контура
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
                cX, cY = 0, 0

        # Распознавание формы
        shape = detect_shapes(contour)

        # Отрисовка контура и названия формы
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Отображение кадра
    cv2.imshow("Shape detection", frame)

    # Для выхода нажмите клавишу 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()