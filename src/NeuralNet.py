import tensorflow as tf
from numpy.typing import NDArray


class NeuralNet:
    def __init__(self, path: str = None, layers: list[tuple[int, str]] = None, optimizer: str = None,
                 loss: str = None, lr: float = None, metrics: list[str] = None):

        if path:
            self.model = tf.keras.models.load_model(path)
        else:
            model = tf.keras.models.Sequential()
            for layer, activation_func in layers:
                model.add(tf.keras.layers.Dense(layer, activation=activation_func))

            if lr:
                if optimizer == 'rmsprop':
                    tf_optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
                elif optimizer == 'adam':
                    tf_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                elif optimizer == 'adagrad':
                    tf_optimizer = tf.keras.optimizers.Adagrad(learning_rate=lr)
                else:
                    tf_optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
                model.compile(optimizer=tf_optimizer, loss=loss, metrics=metrics)
            else:
                model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            self.model = model

    def predict(self, input_data: NDArray, verbose: int = 0):
        """
        Predicts output from input using the neural network

        :param input_data: input data
        :param verbose: verbosity level
        :return: prediction
        """
        return self.model.predict(input_data, verbose=verbose)

    def fit(self, input_data: NDArray, target_data: NDArray, epochs: int, verbose: int = 0):
        """
        Fits the neural network using input data and target data

        :param input_data: input data
        :param target_data: target data
        :param epochs: epochs to train
        :param verbose: verbosity level
        """
        self.model.fit(input_data, target_data, epochs=epochs, verbose=verbose)

    def save(self, path: str, overwrite: bool = False):
        """
        Saves the weights of the neural network to the specified path
        :param path: the path to save the weights
        :param overwrite: whether to overwrite the existing weights
        """
        self.model.save(path, overwrite=overwrite)
