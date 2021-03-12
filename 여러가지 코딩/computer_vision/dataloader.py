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

    
    def food_tf_dataset(self, tfr_size):

        train_size = int(float(self.train_valid_rate[0]) * tfr_size)
        val_size = int(float(self.train_valid_rate[1]) * tfr_size)

        dataset = self.parsed_image_dataset
        dataset = dataset.shuffle(30000)
        # train
        train_ds = dataset.take(train_size)
        train_ds = train_ds.map(self._decode_img)
        train_ds = train_ds.batch(self.batch_size)
        train_ds = train_ds.repeat()
        train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        valid_ds = dataset.skip(train_size)
        valid_ds = dataset.take(val_size)
        valid_ds = valid_ds.map(self._decode_img)
        valid_ds = valid_ds.batch(self.batch_size)

        return train_ds, valid_ds