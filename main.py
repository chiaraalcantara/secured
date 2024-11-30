from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
from face_detection.image_face_detection import detect_faces_in_image

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/detect-faces")
async def detect_faces(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Use face detection function
    processed_img, faces_found = detect_faces_in_image(img)

    # Convert processed image to base64
    _, buffer = cv2.imencode('.jpg', processed_img)
    processed_img_str = base64.b64encode(buffer).decode()
    
    return {
        "image": processed_img_str,
        "faces_found": faces_found
    }