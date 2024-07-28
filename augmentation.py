import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

# TODO: seems like this function and augment_images have a lot of overlap, can we generalize? 
def augment_image_infolder(folder : str):
    """
    Purpose: 
        - augments images in the same folder 
        - performs transformations (rotation, horizontal shift, vertical shift, zoom, shear, reflection, brightness)

    Args: 
        - folder (str) : image source and download path 

    Raises:
        - FileNotFoundError : if image is not found from source (folder)

    """

    # generate batches of tensor image data (define augmentation)
    datagen = ImageDataGenerator(
        # range (in degrees) within which to randomly rotate images
        rotation_range=80,
        # shift images horizontally by 30% 
        width_shift_range=0.3,
        # shift images vertically by 30%
        height_shift_range=0.3,
        # shear angle in counter-clockwise direction in degree changes by 10% 
        shear_range=0.1,
        # images will be zoomed in or out by up to 30%
        zoom_range=0.3,
        # flips horizonally 
        horizontal_flip=True,
        # adjusts brightness 
        brightness_range=[0.5, 1.5],
        # how to fill when pictures are transformed beyond pixel boundaries 
        fill_mode='nearest'
    )
    
    img_file = None

    # Look for the image folder
    for entry in os.listdir(folder):
        c_path = os.path.join(folder, entry)
        # TODO: can we generalize this to work with png files as well? 
        if os.path.isfile(c_path) and (entry.lower().endswith('.jpeg') or entry.lower().endswith('.jpg')):
            img_file = c_path
            break

    # TODO: change this to try catch
    if img_file is None:
        print("Could not find image file in folder that was given")
        return
    
    # Convert image to numpy array and add batch dimension 
    img = load_img(img_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Generate and save augmented images
    for i, batch in enumerate(datagen.flow(x, batch_size=1, save_to_dir=folder, save_prefix='augment', save_format='jpeg')):
        if i == 5: break

# TODO: should we keep this? or is it too specific to that one dataset 
def adult_augmentation(data_path):
    datagen = ImageDataGenerator(
        rotation_range=80,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.1,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.5, 1.5],
        fill_mode='nearest')
    
    entries = os.listdir(data_path)
    folders = [entry for entry in entries if os.path.isdir(os.path.join(data_path, entry))]
    for folder in folders:
        print(os.path.join(data_path, folder))
        for entry in os.listdir(os.path.join(data_path, folder)):
            path = os.path.join(data_path, folder, entry)
            if os.path.isfile(path) and (entry.lower().endswith('.jpeg') or entry.lower().endswith('.jpg')):
                print(path)
                
                img = load_img(path)
                x = img_to_array(img)
                x = x.reshape((1,) + x.shape)
                i = 0
                for batch in datagen.flow(x, batch_size=1, save_to_dir=os.path.join(data_path, folder), save_prefix='augment', save_format='jpeg'):
                    i = i + 1
                    if i >= 5:
                        break
                break

# TODO: Fix documentation
def augment_images(image_folder, send_folder, n):
    """
    image_folder: <- folder with images that you'd like to augment
    send_folder: <- folder that you'd like the new images to be sent to
    n <- number of transformed images per one original image
    You should be able to augment inside the same folder
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
    img_file = None

    for entry in os.listdir(image_folder): # Look for the image folder
        img_file = os.path.join(image_folder, entry)
        if os.path.isfile(img_file) and (entry.lower().endswith('.jpeg') or entry.lower().endswith('.jpg')):
            img = load_img(img_file)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=send_folder, save_prefix='augment', save_format='jpeg'):
                i = i + 1
                if i >= n: # n Transformations
                    break

def augment_image(image_file, send_folder, n):
    """
    image_file: <- image that you'd like to augment
    send_folder: <- folder that you'd like the new images to be sent to
    n <- number of transformed images per original image
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

    img = load_img(image_file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=send_folder, save_prefix='augment', save_format='jpeg'):
        i = i + 1
        if i >= n: # n Transformations
            break