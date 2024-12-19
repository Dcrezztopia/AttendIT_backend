import cv2
import os
import numpy as np

# Path ke dataset dan model
DATASET_PATH = "./datasets"
MODEL_PATH = "./models/face.recognizer.yml"

# Fungsi untuk membaca dataset dan menyiapkan data pelatihan
def prepare_training_data(dataset_path):
    faces = []
    labels = []

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Iterasi melalui folder dataset
    for label, folder_name in enumerate(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for image_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, image_name)
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Deteksi wajah
            faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            for (x, y, w, h) in faces_detected:
                face = gray[y:y + h, x:x + w]
                faces.append(face)
                labels.append(label)

    return faces, labels

# Fungsi untuk melatih model
def train_model():
    faces, labels = prepare_training_data(DATASET_PATH)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)
    print(f"Model telah dilatih dan disimpan di {MODEL_PATH}")

# Fungsi untuk menambahkan dataset foto secara langsung
def capture_dataset_for_training(name):
    cap = cv2.VideoCapture(0)
    
    # Membuat folder untuk anggota kelompok jika belum ada
    if not os.path.exists(f"{DATASET_PATH}/{name}"):
        os.makedirs(f"{DATASET_PATH}/{name}")
    
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            # Menampilkan gambar yang sedang diambil
            cv2.imshow("Capture Photo", frame)

            # Menyimpan foto langsung di folder anggota sesuai nama
            filename = f"{DATASET_PATH}/{name}/{name}_{i:04d}.jpg"
            cv2.imwrite(filename, frame)
            
            # Jika tombol 'q' ditekan atau sudah mencapai 71 gambar, berhenti
            if cv2.waitKey(100) == ord('q') or i == 69:
                break
            i += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"Dataset foto untuk {name} telah selesai disimpan.")

if __name__ == "__main__":
    # Menambahkan dataset foto anggota kelompok (misalnya 'Pascalis')
    name = "Pascalis"  # Anda bisa mengganti dengan nama lain untuk anggota lain
    capture_dataset_for_training(name)
    
    # Melatih model setelah foto disiapkan
    train_model()
