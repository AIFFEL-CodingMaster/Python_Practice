import os
import re
from glob import glob
from PIL import Image
import tensorflow as tf 

class MakeTFRecord:
    
    IMG_SIZE = 224

    def __init__(self, data_list, tfr_path, data_class):
        self.data_list = data_list 
        self.tfr_path = tfr_path 
        self.data_class = data_class

    def _make_tf_writer(self):
        writer = tf.io.TFRecordWriter(self.tfr_path)
        return writer

    # The following functions can be used to convert a value to a type compatible
    # with tf.Example.
    @staticmethod
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    @staticmethod
    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _make_tfrecord(self):
        writer = self._make_tf_writer()
        n = 0
        for data in self.data_list:
            image = Image.open(data)
            image = image.resize((self.IMG_SIZE, self.IMG_SIZE))
            # tf record 는 byte로 되어 있음
            image_to_byte = image.tobytes()

            basename = os.path.basename(data)
            label = os.path.splitext(basename)[0]
            label = re.sub("_\d+","",label)
            label_num = self.data_class[label]

            example = tf.train.Example(features=tf.train.Features(feature={
                "image" : self._bytes_feature(image_to_byte),
                "label" : self._int64_feature(label_num)
            }))

            writer.write(example.SerializeToString())
            n += 1
        writer.close()
        print(f"{n}개의 데이터 TFRecord 완성")
    
    @classmethod
    def change_img_size(cls, image_size):
        cls.IMG_SIZE = image_size
    
    def __call__(self):
        print("tfrecord 만들기 시작!")
        self._make_tfrecord()
