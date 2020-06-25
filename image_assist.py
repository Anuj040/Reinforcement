"""
auxillary functions for loading and preprocessing the data

"""
import numpy as np
import random
from keras.preprocessing import image

def load_images_in_data_set(data_set_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + data_set_name + '.txt'
    f = open(file_path)
    image_names = f.readlines()
    image_names = [x.strip('\n') for x in image_names]
    image_names = random.sample(image_names, 4000)
    if data_set_name.startswith('aeroplane') | data_set_name.startswith('bird') | data_set_name.startswith('cow'):
        return sorted([x.split(None, 1)[0] for x in image_names])
    else:
        return sorted([x.split('\n')[0] for x in image_names])

def get_all_images(image_names, path_voc):
    images = []
    for j in range(np.size(image_names)):
        image_name = image_names[0][j]
        string = path_voc + '/JPEGImages/' + str(image_name) + '.jpg'
        images.append(image.load_img(string, False))
    return images

def load_image_labels(image_names, dataset_name, path_voc):
    file_path = path_voc + '/ImageSets/Main/' + dataset_name + '.txt'
    f = open(file_path)
    image_labels = f.readlines()
    image_labels = [x.split(None, 1)[1] for x in image_labels if x.split(None, 1)[0] in image_names]
    image_labels = [x.strip('\n') for x in image_labels]
    return image_labels

# def mask_image_with_mean_background(mask_object_found, image):
#     new_image = image
#     size_image = np.shape(mask_object_found)
    