import tensorflow as tf 


class TFRecordLoader:
    
    def __init__(self, tfrecord_path, img_size, n_class):
        self.tfrecord = tfrecord_path
        self.img_size = img_size
        self.n_class = n_class

    ## tfrecord file을 data로 parsing해주는 function
    def _parse_function(tfrecord_serialized):
        features={'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([], tf.int64)
                }
        parsed_features = tf.io.parse_single_example(tfrecord_serialized, features)
        
        image = tf.io.decode_raw(parsed_features['image'], tf.uint8)
        image = tf.reshape(image, [self.img_size, self.img_size, 3])
        # image = tf.cast(image, tf.float32)/255. 

        label = tf.cast(parsed_features['label'], tf.int64)
        label = tf.one_hot(label, self.n_class)

        return image, label

    def make_dataset(self):
        dataset = tf.data.TFRecordDataset(self.tfrecord)
        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        buffer_size = len(list(dataset))
        dataset = dataset.shuffle(buffer_size).prefetch(tf.data.experimental.AUTOTUNE).batch(32)
        return dataset
    
    def __call__(self):
        return self.make_dataset()