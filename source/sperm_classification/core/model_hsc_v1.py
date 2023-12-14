from keras import Input
from keras.layers import Conv2D, Dropout, GlobalAveragePooling2D
from keras.layers import Dense, BatchNormalization, MaxPooling2D, concatenate, Activation, Flatten
# from keras.layers.pooling import AveragePooling2D
from keras.layers import AveragePooling2D
from keras.models import Model
from keras.utils import plot_model
import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
print('ROOT = ', ROOT)
from core.utils_activation import activation_functions


def block_stem(x, filter_cnv, pool_size=(2, 2), kernel_size=7, strides=1, activation='relu', padding='same',
               name="Block_Stem"):
    conv_stem = Conv2D(filter_cnv, kernel_size=kernel_size, strides=strides, activation=activation_functions[activation], padding=padding)(x)
    conv_stem = MaxPooling2D(pool_size=pool_size)(conv_stem)
    conv_stem = Conv2D(filter_cnv * 2, kernel_size=kernel_size // 2, strides=strides, activation=activation_functions[activation],
                       padding=padding)(conv_stem)
    conv_stem = MaxPooling2D(pool_size=pool_size, name=name)(conv_stem)
    return conv_stem


def block_conv(x, filter_block, activation='relu', padding='same', name="block_conv"):
    print(x.shape)
    conv_a = Conv2D(filter_block, kernel_size=1, strides=2, activation=activation_functions[activation], padding=padding)(x)
    conv_a = Conv2D(filter_block // 2, kernel_size=7, strides=1, activation=activation_functions[activation], padding=padding)(conv_a)
    conv_a = BatchNormalization()(conv_a)

    conv_b = Conv2D(filter_block, kernel_size=1, strides=1, activation=activation_functions[activation], padding=padding)(x)
    conv_b = Conv2D(filter_block // 2, kernel_size=5, strides=2, activation=activation_functions[activation], padding=padding)(conv_b)
    conv_b = BatchNormalization()(conv_b)

    conv_c = AveragePooling2D(pool_size=(2, 2))(x)
    conv_c = Conv2D(filter_block // 2, kernel_size=3, strides=1, activation=activation_functions[activation], padding=padding)(conv_c)
    conv_c = BatchNormalization()(conv_c)

    conv_d = AveragePooling2D(pool_size=(2, 2))(x)
    conv_d = BatchNormalization()(conv_d)

    conv_concat = concatenate([conv_a, conv_b, conv_c, conv_d])
    output_block = Activation(activation, name=name)(conv_concat)
    return output_block
    pass


def block_identity(x, filter_block, kernel_size_a=3, kernel_size_b=1, stride=1, activation='relu', padding='same',
                   name="block_identity"):
    conv_a = Conv2D(filter_block, kernel_size=kernel_size_a, strides=stride, activation=activation_functions[activation], padding=padding)(x)
    conv_a = BatchNormalization()(conv_a)

    conv_b = Conv2D(filter_block, kernel_size=kernel_size_b, strides=stride, activation=activation_functions[activation], padding=padding)(x)
    conv_b = BatchNormalization()(conv_b)

    conv_concat = concatenate([conv_a, conv_b, x])
    output_block = Activation(activation, name=name)(conv_concat)
    return output_block


def block_rep_residual(x_input, filter_conv=32, filter_id_a=64, filter_id_b=128, activation_block='relu',
                       name='rep_residual'):
    x_conv_a = block_conv(x=x_input, filter_block=filter_conv, activation=activation_block,
                          padding='same', name="block_conv_a_" + name)

    x_id_a = block_identity(x_conv_a, filter_block=filter_id_a, kernel_size_a=5, kernel_size_b=3, stride=1,
                            activation=activation_block,
                            padding='same', name="block_identity_a_" + name)

    x_concat_a = concatenate([x_conv_a, x_id_a], name="concat_a_" + name)

    x_id_b = block_identity(x_concat_a, filter_block=filter_id_b, kernel_size_a=3, kernel_size_b=1, stride=1,
                            activation=activation_block, padding='same',
                            name="block_identity_b_" + name)
    x_rep_residual = concatenate([x_conv_a, x_id_a, x_id_b], name=name)
    return x_rep_residual


def created_model_hsc_01(input_shape, number_class=2, activation_dense='softmax', activation_block='relu'):
    input_layer = Input(shape=input_shape)

    x_stem = block_stem(x=input_layer, filter_cnv=16, pool_size=(2, 2), kernel_size=7, strides=1,
                        activation=activation_block, padding='same', name="Block_Stem")

    x_rep_a = block_rep_residual(x_input=x_stem, filter_conv=64, filter_id_a=64, filter_id_b=64,
                                 activation_block=activation_block, name='rep_residual_a')

    x_pool_a = MaxPooling2D(pool_size=(2, 2))(x_rep_a)

    x_rep_b = block_rep_residual(x_input=x_pool_a, filter_conv=128, filter_id_a=128, filter_id_b=128,
                                 activation_block=activation_block, name='rep_residual_b')

    x = GlobalAveragePooling2D()(x_rep_b)
    x = Dropout(0.5)(x)
    x = Dense(number_class, activation=activation_dense)(x)
    return Model(inputs=input_layer, outputs=x)


# model = created_model_hsc_01(input_shape=(40, 40, 3), number_class=4,
#                              activation_dense='softmax', activation_block='LeakyReLU')
# model.summary(show_trainable=True)
