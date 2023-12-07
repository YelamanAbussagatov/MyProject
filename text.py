import cv2
import pytesseract

# Установите путь к исполняемому файлу Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR'

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
while True:

    # Считывание кадра из видеопотока
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Распознавание текста с помощью Tesseract OCR
    text = pytesseract.image_to_string(gray)

    # Вывод распознанного текста на экран
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Webcam Text Recognition', frame)

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()