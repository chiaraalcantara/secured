import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

def augment_image(image_files, send_folder, n):
    """
    image_file: <- list images that you'd like to augment
    send_folder: <- folder that you'd like the new images to be sent to
    n <- number of transformed images per original image
    TODOs:
     - allow png's
    """
    datagen = ImageDataGenerator(
        rotation_range=80,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest')
    
    for image_file in image_files:
        try:
            img = load_img(image_file)
        except:
            print(f"{image_file} could not be loaded")
            continue

        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=send_folder, save_prefix='augment', save_format='jpeg'):
            i = i + 1
            if i >= n: # n Transformations
                break