import cv2
import qrcode
# Функция для распознавания QR-кодов на изображении
def read_qr_code(image_path):
    img = cv2.imread(image_path)
    detector = cv2.QRCodeDetector()
    val, pts, qr_code = detector.detectAndDecode(img)

    if val:
        return val
    else:
        return None

# Основная функция для захвата видеопотока с веб-камеры

def capture_qr_code():
    cap = cv2.VideoCapture(0)  # 0 - индекс веб-камеры
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        detector = cv2.QRCodeDetector()
        val, pts, qr_code = detector.detectAndDecode(frame)
        if val:
            print("Распознанный QR-код: ", val)
            break

        cv2.imshow("QR Code Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    # Вызов функции захвата видеопотока с веб-камеры
    capture_qr_code()