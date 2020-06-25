import xml.etree.ElementTree as ET
import numpy as np


def get_bb_gt(xml_name, voc_path):
    string = voc_path + '/Annotations/' + xml_name + '.xml'
    tree = ET.parse(string)
    root = tree.getroot()
    names = []
    x_min = []
    x_max = []
    y_min = []
    y_max = []

    for child in root:
        if child.tag == 'object':
            for child2 in child:
                if child2.tag == 'name':
                    names.append(child2.text)
                elif child2.tag == 'bndbox':
                    for child3 in child2:
                        if child3.tag == 'xmin':
                            x_min.append(child3.text)
                        elif child3.tag == 'xmax':
                            x_max.append(child3.text)
                        elif child3.tag == 'ymin':
                            y_min.append(child3.text)
                        elif child3.tag == 'ymax':
                            y_max.append(child3.text)
    category_and_bb = np.zeros([np.size(names), 5], dtype=np.int32)
    for i in range(np.size(names)):
        category_and_bb[i][0] = get_id_of_class_name(names[i])
        category_and_bb[i][1] = x_min[i]
        category_and_bb[i][2] = x_max[i]
        category_and_bb[i][3] = y_min[i]
        category_and_bb[i][4] = y_max[i]
    return category_and_bb

def get_id_of_class_name (class_name):
    class_id = {'aeroplane': 1,
                'bicycle': 2,
                'bird': 3,
                'boat': 4,
                'bottle': 5,
                'bus': 6,
                'car': 7,
                'cat': 8,
                'chair': 9,
                'cow': 10,
                'diningtable': 11,
                'dog': 12,
                'horse': 13,
                'motorbike': 14,
                'person': 15,
                'pottedplant': 16,
                'sheep': 17,
                'sofa': 18,
                'train': 19,
                'tvmonitor': 20
                }
    return class_id[class_name]

def generate_bounding_box(annotation, image_shape):
    length_annotation = annotation.shape[0]
    masks = np.zeros([image_shape[0], image_shape[1], length_annotation])
    for i in range(length_annotation):
        masks[annotation[i,3]:annotation[i,4], annotation[i,1]:annotation[i,2], i] = 1
    return masks

