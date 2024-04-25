import tensorflow as tf


class NeuralNet:
    def __init__(self, layers: list[tuple[int, str]], optimizer: str, loss: str, metrics: list[str] = None):
        model = tf.keras.models.Sequential()
        for layer, activation_func in layers:
            model.add(tf.keras.layers.Dense(layer, activation=activation_func))
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def predict(self, input_data, verbose=0):
        """
        Predicts output from input using the neural network

        :param input_data: input data
        :param verbose: verbosity level
        :return: prediction
        """
        return self.model.predict(input_data, verbose=verbose)

    def fit(self, input_data, target_data, epochs, verbose=0):
        """
        Fits the neural network using input data and target data

        :param input_data: input data
        :param target_data: target data
        :param epochs: epochs to train
        :param verbose: verbosity level
        """
        self.model.fit(input_data, target_data, epochs=epochs, verbose=verbose)

    def save_weights(self, path, overwrite=False):
        """
        Saves the weights of the neural network to the specified path
        :param path: the path to save the weights
        :param overwrite: whether to overwrite the existing weights
        """
        self.model.save_weights(path, overwrite=overwrite)
