from keras.callbacks import ModelCheckpoint, EarlyStopping


def train(name, model, sentences, labels, validation_set=0.2, n_epochs=10, allow_stop=False):
    """
    Realiza el entrenamiento completo de un modelo (con un conjunto de datos de validación).
    """
    # Separamos el conjunto de datos.
    X_train, y_train = sentences, labels
    # Creamos la lista de CallBacks.
    callbacks_list = []
    # Definimos un Checkpoint.
    filepath = f'Checkpoint/checkpoint_{name}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', mode='max', save_best_only=True, verbose=0)
    callbacks_list.append(checkpoint)
    # Definimos un Early Stopping.
    if allow_stop:
        early_stop = EarlyStopping(monitor='val_accuracy', mode='max', restore_best_weights=True, patience=1, verbose=1)
        callbacks_list.append(early_stop)
    # Realizamos el entrenamiento (junto con la validación).
    history = model.fit(X_train, y_train, validation_split=validation_set, callbacks=callbacks_list, epochs=n_epochs, verbose=1)
    return history


def cross_validation(model, filepath, cv_split, sentences, labels, n_epochs=1):
    """
    Realiza la validación cruzada de un modelo.
    """
    splits = 0
    mean_loss = 0.0
    mean_accuracy = 0.0
    for train_index, val_index in cv_split:
        # Separamos el conjunto de datos.
        X_train = sentences[train_index]
        X_val = sentences[val_index]
        y_train = labels[train_index]
        y_val = labels[val_index]
        # Realizamos el entrenamiento y la evaluación.
        model.fit(X_train, y_train, epochs=n_epochs, verbose=1)
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        # Reiniciamos los pesos del modelo.
        model.load_weights(filepath)
        # Acumulamos la pérdida y la exactitud.
        splits += 1
        mean_loss += loss
        mean_accuracy += accuracy
    # Obtenemos el promedio de pérdida y exactitud.
    mean_loss /= splits
    mean_accuracy /= splits
    return mean_loss, mean_accuracy

