import math
import tensorflow as tf 


class TFRecordLoader:
    
    def __init__(self, tfrecord_path, img_size, n_class, train_size_rate, batch_size):
        self.tfrecord = tfrecord_path
        self.img_size = img_size
        self.n_class = n_class
        self.train_size_rate = train_size_rate
        self.batch_size = batch_size

    ## tfrecord file을 data로 parsing해주는 function
    def _parse_function(self, tfrecord_serialized):
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
        dataset = dataset.map(self._parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        train_size = int(float(self.train_size_rate * len(list(dataset))))
        val_size = int(float((1 - self.train_size_rate) * len(list(dataset))))

        buffer_size = len(list(dataset))
        dataset = dataset.shuffle(buffer_size)
        train = dataset.take(train_size)
        train = train.batch(self.batch_size)
        train = train.repeat()
        train = train.prefetch(tf.data.experimental.AUTOTUNE)

        dataset = dataset.skip(train_size)
        valid = dataset.take(val_size)
        valid = valid.batch(self.batch_size)

        steps = math.floor(buffer_size / self.batch_size)


        return train, valid ,steps
    
    def __call__(self):
        return self.make_dataset()
