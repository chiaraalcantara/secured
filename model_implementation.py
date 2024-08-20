from tensorflow.keras.models import load_model
import numpy as np


# Load the pre-trained FaceNet model
model = load_model('facenet_keras.h5')

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
def detect_faces(image_path : str):
    """
    Purpose: 
        - detects faces in an image from local directory 

    Args: 
        - image_path (str): file path to image 

    Raises:
        - FileNotFoundError : if image is not found from source (folder)

    """

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

