from keras import backend as K
from keras.layers import ReLU, ELU, LeakyReLU
from keras.utils import get_custom_objects
from keras.layers import Activation


def seg_relu(x):
    return K.switch(x > 0, x, K.softsign(x))


get_custom_objects().update({'seg_relu': Activation(seg_relu)})
activation_functions = {
    'relu': ReLU(),
    'ELU': ELU(),
    'LeakyReLU': LeakyReLU(),
    'seg_relu': 'seg_relu'
}
