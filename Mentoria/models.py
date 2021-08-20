from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization


class NN_Baseline_Model():
    def __init__(self, embedding_layer, classes):
      self.name = 'Neural Network Baseline'

      self.model = Sequential()
      # Definición arbitraria del modelo.
      self.model.add(embedding_layer)
      self.model.add(Dense(256, activation='sigmoid'))
      self.model.add(Dense(128, activation='sigmoid'))
      self.model.add(Flatten())
      self.model.add(Dense(len(classes), activation='softmax'))

      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def model_summary(self):
        self.model.summary()


class NN_Dropout_Model():
    def __init__(self, embedding_layer, classes):
      self.name = 'Neural Network Dropout'

      self.model = Sequential()
      # Definición arbitraria del modelo.
      self.model.add(embedding_layer)
      self.model.add(Dense(256, activation='relu'))
      self.model.add(Dropout(0.2))
      self.model.add(Dense(128, activation='relu'))
      self.model.add(Dropout(0.2))
      self.model.add(Flatten())
      self.model.add(Dense(len(classes), activation='softmax'))

      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def model_summary(self):
        self.model.summary()


class NN_Batch_Model():
    def __init__(self, embedding_layer, classes):
      self.name = 'Neural Network Batch'

      self.model = Sequential()
      # Definición arbitraria del modelo.
      self.model.add(embedding_layer)
      self.model.add(Dense(256, activation='sigmoid'))
      self.model.add(BatchNormalization())
      self.model.add(Dense(128, activation='sigmoid'))
      self.model.add(BatchNormalization())
      self.model.add(Flatten())
      self.model.add(Dense(len(classes), activation='softmax'))

      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def model_summary(self):
        self.model.summary()

