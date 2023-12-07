import cv2
import pytesseract

# Установите путь к исполняемому файлу Tesseract OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Захват видео с веб-камеры
cap = cv2.VideoCapture(0)
while True:
    # Считывание кадра из видеопотока
    ret, frame = cap.read()

    # Преобразование кадра в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Применение фильтра Canny для обнаружения границ
    edges = cv2.Canny(gray, 100, 200)

    # Распознавание текста с помощью Tesseract OCR
    text = pytesseract.image_to_string(edges, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')

    # Вывод распознанных цифр на экран
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Webcam Digit Recognition', frame)
    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()