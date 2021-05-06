import tensorflow as tf 


class Optimizer:
    
    def __init__(self, optim, lr=0.01):
        
        if optim =="sgd":
            self.optim = tf.keras.optimizers.SGD(lr=lr)
        elif optim =="adam":
            self.optim = tf.keras.optimizers.Adam(lr=lr)
        else:
            raise ValueError("Errrrrrrror")
class Loss:
    
    def __init__(self, loss):

        if loss == "cc":
            self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif loss == "sc":
            self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            raise ValueError("Errrrrrrror")