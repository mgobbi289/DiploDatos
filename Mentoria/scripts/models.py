from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, LSTM
from keras.regularizers import l2


class NN_Baseline():
    """
    Red Neuronal Básica
    - Un par de capas densas con activación sigmoide.
    """
    def __init__(self, embedding_layer, classes):
        self.name = 'NN_Baseline'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN sin Regularización.
        return False


class NN_Dropout():
    """
    Red Neuronal con Dropout
    - Un par de capas densas con activación relu.
    - Un par de capas dropout con 0.5 de probabilidad.
    """
    def __init__(self, embedding_layer, classes):
        self.name = 'NN_Dropout'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN sin Regularización.
        return False


class NN_BatchNorm():
    """
    Red Neuronal con Normalización
    - Un par de capas densas con activación sigmoide.
    - Un par de capas con batchnormalization.
    """
    def __init__(self, embedding_layer, classes):
        self.name = 'NN_BatchNorm'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(BatchNormalization())
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN sin Regularización.
        return False


class NN_LSTM():
    """
    Red Neuronal con Long Short-Term Memory
    - Un par de capas densas con activación sigmoide.
    - Una capa con lstm.
    """
    def __init__(self, embedding_layer, classes):
        self.name = 'NN_LSTM'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(LSTM(64))
        self.model.add(Dense(256, activation='sigmoid'))
        self.model.add(Dense(128, activation='sigmoid'))
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN sin Regularización.
        return False


class NN_Baseline_Regularized():
    """
    Red Neuronal Básica
    - Un par de capas densas con activación sigmoide.
    - Término de regularización l2 en capas densas.
    """
    def __init__(self, embedding_layer, classes, reg=0.0001):
        self.name = 'NN_Baseline_Regularized'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(Dense(256, kernel_regularizer=l2(reg), activation='sigmoid'))
        self.model.add(Dense(128, kernel_regularizer=l2(reg), activation='sigmoid'))
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN con Regularización.
        return True


class NN_LSTM_Regularized():
    """
    Red Neuronal con Long Short-Term Memory
    - Un par de capas densas con activación sigmoide.
    - Una capa con lstm.
    - Término de regularización l2 en capas densas y en capa lstm.
    """
    def __init__(self, embedding_layer, classes, reg=0.0001):
        self.name = 'NN_LSTM_Regularized'

        self.model = Sequential()
        # Definición arbitraria del modelo.
        self.model.add(embedding_layer)
        self.model.add(LSTM(64, kernel_regularizer=l2(reg)))
        self.model.add(Dense(256, kernel_regularizer=l2(reg), activation='sigmoid'))
        self.model.add(Dense(128, kernel_regularizer=l2(reg), activation='sigmoid'))
        self.model.add(Flatten())
        self.model.add(Dense(len(classes), activation='softmax'))

    def model_compile(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        # En caso de necesitar reiniciar el entrenamiento de forma determinística.
        self.filepath = f'Checkpoint/initial_weights_{self.name}.hdf5'
        self.model.save_weights(self.filepath)

    def model_summary(self):
        self.model.summary()

    def model_allow_stop(self):
        # NN con Regularización.
        return True

