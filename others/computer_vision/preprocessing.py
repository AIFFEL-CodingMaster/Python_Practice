import os
import re
from PIL import Image


def delete_mat(data_list):
    for i, data in enumerate(data_list):
        basename = os.path.basename(data)
        _, file = basename.split(".")
        if file == "mat":
            del data_list[i]
    return data_list

def delete_4_channel(data_list):
    for i ,data in enumerate(data_list):
        image = Image.open(data)
        mode = image.mode
        # mode가 RGB 말고 P도 있음
        if mode != "RGB":
            del data_list[i]
    return data_list

def label_encoding(data_list):
    class_list = []
    for data in data_list:
        basename = os.path.basename(data)
        label = os.path.splitext(basename)[0]
        
        label = re.sub("_\d+", '', label)
        
        if label in class_list:
            continue
        else:
            class_list.append(label)
    class_list = list(set(class_list))
    class_to_index = { cls : i for i, cls in enumerate(class_list) }

    return class_to_index

