import cv2
import os

# Membuat folder 'images' jika belum ada
if not os.path.exists('images'):
    os.makedirs('images')

# Inisialisasi video dan detektor wajah
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        # Menambahkan persegi panjang pada wajah yang terdeteksi
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if len(faces_data) == 0:
            faces_data.append(resized_img)
            break  # Keluar dari loop setelah mengambil satu foto
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q'):
        # Jika tombol 'q' ditekan, simpan gambar dan keluar dari loop
        if len(faces_data) > 0:
            cv2.imwrite(f'images/{name}.jpg', faces_data[0])
        break

video.release()
cv2.destroyAllWindows()
