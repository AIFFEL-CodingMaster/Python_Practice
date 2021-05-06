import math
import pandas as pd 
import matplotlib.pyplot as plt 
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 


def visualize_history(hist):
    fig, axes = plt.subplots(1,2, figsize=(20,12))
    axes[0].set_title("train, validation loss", fontsize=20)
    axes[1].set_title("train, validation accuracy", fontsize=20)
    axes[0].plot(hist["loss"], label="loss")
    axes[0].plot(hist['val_loss'], label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].axes.legend()
    axes[1].plot(hist["accuracy"], label="accuracy")
    axes[1].plot(hist["val_accuracy"], label="val_accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].axes.legend()

class CosineAnnealingScheduler(Callback):
    """
    Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)