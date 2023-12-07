import cv2
import numpy as np

# Загрузите предварительно обученную модель для обнаружения объектов (например, SSD, YOLO, Faster R-CNN)
# Здесь предполагается использование OpenCV для YOLO
net = cv2.dnn.readNet("C://Users//User//Documents//Elaman//yolov3.weights", "C://Users//User//Documents//Elaman//yolov3.cfg")

# Загрузите классы объектов
classes = []
with open("C://Users//User//Documents//Elaman//coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Используйте веб-камеру (замените 0 на номер вашей камеры, если она не является первой)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получите высоту и ширину кадра
    height, width = frame.shape[:2]

    # Преобразуйте кадр в блоб
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    # Загрузите блоб в сеть и получите результаты обнаружения
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Инициализируйте списки для обнаруженных объектов
    class_ids = []
    confidences = []
    boxes = []

    # Проанализируйте результаты обнаружения и отфильтруйте объекты
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # Порог уверенности
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Координаты прямоугольника
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Удалите неоднозначные обнаружения с использованием Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Отобразите результаты на кадре
    for i in indices:
        box = boxes[i]
        x, y, w, h = box
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = (0, 255, 0)  # Зеленый цвет
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Отобразите кадр с результатами
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Освободите ресурсы
cap.release()
cv2.destroyAllWindows()