import os 
import re 
from PIL import Image

# mat 삭제
def delete_mat(data_list):
    for i, data in enumerate(data_list):
        basename = os.path.basename(data)
        _, file = basename.split(".")
        
        if file == "mat":
            del data_list[i]
    return data_list

# 4 channel 삭제 
def delete_4_channel(data_list):
    for i, data in enumerate(data_list):
        image_data = Image.open(data)
        mode = image_data.mode
        
        if mode != "RGB":
            del data_list[i]
    return data_list

# 라벨 인코딩
def label_encoding(data_list):
    # 방법 1
    class_list = []
    for data in data_list:
        basename = os.path.basename(data)
        label = os.path.splitext(basename)[0]
        
        label = re.sub("_\d+", "", label)
        
        if label in class_list:
            continue
        else:
            class_list.append(label)
    class_to_index = { cls : i for i, cls in enumerate(class_list) }
    return class_to_index