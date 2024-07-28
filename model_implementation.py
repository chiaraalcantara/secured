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



def get_face_embedding(face_image : np.ndarray):
    """
    Purpose: 
        - function to extract and normalize face embeddings as 160x160 ndarray 
        - adds batch dimension 
        - feature extraction 

    Args: 
        - face_image (np.ndarray): ndarray of any size 

    Raises:

    """

# def upload_face_embedding(face_embedding : np.ndarray): 
#     """
#     Purpose: 
#         - uploads face embedding to vector db

#     Args: 
#         - face_embedding (np.ndarray): ndarray 

#     Raises:
#         - raise exception if upload is unsuccessful 

#     """


def detect_faces(image_path : str):
    """
    Purpose: 
        - detects faces in an image from local directory (future will get from R2)

    Args: 
        - image_path (str): file path to image 

    Raises:
        - FileNotFoundError : if image is not found from source (folder)

    """


