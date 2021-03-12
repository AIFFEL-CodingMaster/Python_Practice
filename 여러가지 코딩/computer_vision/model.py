import tensorflow as tf 
from tensorflow import keras 
from collections import deque

class ModelSelect:

    def __init__(self, model_name):

        if model_name == "e0":
            self.model = keras.applications.EfficientNetB0(
                include_top=False,
                pooling="avg"
            )
        elif model_name == "mobilev2":
            self.model = keras.applications.MobileNetV2(
                include_top=False,
                pooling="avg"
            )

class MakeModel(ModelSelect):
    
    def __init__(self, model_name):
        super(MakeModel, self).__init__(model_name)
    
    def add_dense_layer(self, nums ,activation="n"):
        if activation == "n":    
            return keras.layers.Dense(nums, activation=None)
        else:
            return keras.layers.Dense(nums, activation=activation)
    
    def add_batch_norm(self):
        return keras.layers.BatchNormalization()
    
    def make_model_with_FCL(self, img_size ,*args):
        temp = deque([i for i in args[0]])
        args = temp
        
        inputs = keras.layers.Input((img_size, img_size, 3))
        self.model = self.model(inputs)
        while args:
            self.model = args[0](self.model)
            args.popleft()
        model = keras.Model(inputs, self.model)
        return model