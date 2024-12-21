from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize face cascade classifier
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
print(f"Loading cascade from: {CASCADE_PATH}")
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# Initialize LBPH recognizer
MODEL_PATH = "./models/face.recognizer.yml"
print(f"Loading model from: {MODEL_PATH}")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_PATH)

LABELS_TO_NAMES = {
    0: "Dela",
    1: "Kinata",
    2: "Mulki",
    3: "Pascalis",
}

def recognize_face(image):
    try:
        # Convert PIL Image to OpenCV format
        open_cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        print("Image converted to OpenCV format")
        
        # Save the received image for debugging
        debug_path = "debug_received.jpg"
        cv2.imwrite(debug_path, open_cv_image)
        print(f"Saved debug image to {debug_path}")
        
        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        print(f"Converted to grayscale. Shape: {gray.shape}")
        
        # Try different parameters for face detection
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Face detection completed. Found {len(faces)} faces")
        
        if len(faces) == 0:
            # Try with more lenient parameters
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=3,
                minSize=(20, 20)
            )
            print(f"Retried with lenient parameters. Found {len(faces)} faces")

        results = []
        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            
            # Save detected face for debugging
            face_path = f"debug_face_{len(results)}.jpg"
            cv2.imwrite(face_path, face)
            print(f"Saved detected face to {face_path}")
            
            label, confidence = recognizer.predict(face)
            name = LABELS_TO_NAMES.get(label, "Unknown")
            results.append({
                "predicted_name": name,
                "confidence": float(confidence),
                "face_location": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
            })
            print(f"Detected face: {name} with confidence: {confidence}")

        return results
    except Exception as e:
        print(f"Error in recognize_face: {e}")
        import traceback
        traceback.print_exc()
        raise

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):
    try:
        print(f"\n=== New Recognition Request ===")
        print(f"Received file: {file.filename}")
        print(f"File content type: {file.content_type}")
        
        contents = await file.read()
        print(f"File size: {len(contents)} bytes")
        
        image = Image.open(BytesIO(contents))
        print(f"Image size: {image.size}")
        
        results = recognize_face(image)
        print(f"Recognition results: {results}")
        return {"results": results}
    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}