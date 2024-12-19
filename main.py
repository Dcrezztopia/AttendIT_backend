from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

MODEL_PATH = "./models/face.recognizer.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

# Menggunakan file haarcascade yang disediakan oleh OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

LABELS_TO_NAMES = {
    0: "Dela",
    1: "Kinata",
    2: "Mulki",
    3: "Pascalis",
    # Tambahkan label lainnya sesuai dataset Anda
}

# Fungsi untuk mengenali wajah dari gambar
def recognize_face(image):
    gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)
        name = LABELS_TO_NAMES.get(label, "Unknown")
        results.append({"label": label, "name": name, "confidence": confidence})

    return results

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read()))
    results = recognize_face(image)
    print(f"File received: {file.filename}")
    return {"results": results}
