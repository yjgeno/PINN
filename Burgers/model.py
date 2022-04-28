import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


# solution neural network
def neural_net(layer_sizes = [2] + 8*[20] + [1]):
    model = Sequential()
    model.add(layers.InputLayer(input_shape=(layer_sizes[0],)))
    for width in layer_sizes[1:-1]:
        model.add(layers.Dense(
            width, activation=tf.nn.tanh,
            kernel_initializer="glorot_normal"))
    model.add(layers.Dense(
            layer_sizes[-1], activation=None,
            kernel_initializer="glorot_normal"))
    return model


if __name__ == '__main__':
    import sys
    model = neural_net()
    sys.exit(model.summary())