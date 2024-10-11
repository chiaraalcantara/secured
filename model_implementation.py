from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import joblib
import os
import sys
sys.path.append(r'C:\Users\sahil\Documents')
sys.path.append(r'C:\Users\sahil\Documents\yolov7')
from yolov7.models.experimental import attempt_load
from insightface.app import FaceAnalysis
from utils.general import non_max_suppression
import cv2
import torch
from torchvision import transforms
# Load the pre-trained FaceNet model
# model = load_model('facenet_keras.h5')
YOLOV7_WEIGHTS = r"C:\Users\sahil\Documents\yolov7\yolov7.pt"
from typing import Tuple

# def get_picture(file_path : str): 
#     """
#     Purpose: 
#         - takes a picture for testing purposes

#     Args: 
#         - file_path (str) : download path of picture 

#     Raises:
#         - error if picture is not taken 

#     """


# def upload_face_embedding(face_embedding : np.ndarray): 
#     """
#     Purpose: 
#         - uploads face embedding to vector db

#     Args: 
#         - face_embedding (np.ndarray): ndarray 

#     Raises:
#         - raise exception if upload is unsuccessful 

#     """

"""
Feel free to change function signatures if required to complete these funcitons! 

"""

# suggested model: yolov7
def detect_faces(image_path : str, conf_thres: int = 0.4) -> Tuple[bool, int, int, int, int]:
    """
    Purpose: 
        - detects faces in an image from local directory 

    Args: 
        - image_path (str): file path to image
        - conf_thres (int): confidence threshold

    Raises:
        - FileNotFoundError : if image is not found from source (folder)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    model = attempt_load(YOLOV7_WEIGHTS, map_location='cpu')
    model.eval()
    img = Image.open(image_path)
    transform = transforms.Compose([transforms.ToTensor()])
    with torch.no_grad():
        results = model(transform(img.resize((640, 640))).unsqueeze(0))
    predictions = results[0][0]  # Extract the first element of the results tuple

    predictions = non_max_suppression(results[0], conf_thres)
    x1, y1, x2, y2 = 0, 0, 0, 0
    face_detected = False
    for pred in predictions[0]:
        if pred[5] == 0:
            face_detected = True
            x1, y1, x2, y2 = pred[0], pred[1], pred[2], pred[3]
    
    return face_detected, x1, y1, x2, y2 

# suggested model: arcface 
def get_face_embedding(image_path : str):
    """
    Purpose: 
        - function to extract face from image and normalize face embeddings from an image as 160x160 ndarray 
        - adds batch dimension 
        - feature extraction 

    Args: 
        - image_path (str): file path to image 

    Raises:
        - FileNotFoundError : if image is not found from source (folder)

    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    app = FaceAnalysis()
    app.prepare(ctx_id=0)  # ctx_id=0 for CPU, set to a GPU ID for GPU usage
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image_rgb)
    if len(faces) == 0:
        raise ValueError("No face detected in the image.")
    # Assume we only care about the first detected face
    face = faces[0].embedding
    # Normalize the embedding
    face_embedding = np.array(face)
    # Ensure the embedding is in the shape (1, 512)
    face_embedding = face_embedding.reshape(-1)
    return face_embedding

