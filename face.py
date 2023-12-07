import cv2
# Загрузка каскада Хаара для распознавания лиц
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Захват видеопотока с веб-камеры
cap = cv2.VideoCapture(0)

while True:
    # Чтение кадра из видеопотока
    ret, frame = cap.read()
    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Распознавание лиц на кадре
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Отрисовка прямоугольников вокруг распознанных лиц
    for (x, y, w, h) in faces:        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
    # Отображение кадра с распознанными лицами
    cv2.imshow('Face Detection', frame)
    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Освобождение ресурсовcap.release()
cv2.destroyAllWindows()