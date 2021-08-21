

def train(model, training_data, validation_data, n_epochs=1):
    """
    Realiza el entrenamiento de un modelo (con un conjunto de datos de validación).
    """
    # Separamos el conjunto de datos.
    X_train, y_train = training_data
    X_val, y_val = validation_data
    # Realizamos el entrenamiento y la validación.
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=n_epochs, verbose=1)
    return history


def cross_validation(model, cv_split, sentences, labels, n_epochs=1):
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
        # Acumulamos la pérdida y la exactitud.
        splits += 1
        mean_loss += loss
        mean_accuracy += accuracy
    # Obtenemos el promedio de pérdida y exactitud.
    mean_loss /= splits
    mean_accuracy /= splits
    return mean_loss, mean_accuracy
