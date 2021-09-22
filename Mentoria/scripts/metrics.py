import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score


def plot_accuracy_and_loss(history, name):
    """
    Gráfica las curvas de accuracy y loss para los conjuntos
    de datos de entrenamiento y validación.
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    # Accuracy
    ax = axes[0]
    ax.title.set_text(f'{name} Accuracy')
    ax.plot(history.history['accuracy'])
    ax.plot(history.history['val_accuracy'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.legend(['training', 'validation'], loc='upper left')
    # Loss
    ax = axes[1]
    ax.title.set_text(f'{name} Loss')
    ax.plot(history.history['loss'])
    ax.plot(history.history['val_loss'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.legend(['training', 'validation'], loc='upper left')
    # Plot
    fig.tight_layout()


def show_balanced_accuracy(name, model, sentences, labels, df_test):
    """
    Calcula la balanced accuracy para el conjunto de datos de evaluación.
    """
    print(f'{name}')

    probabilities = model.predict(sentences, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    print(f'Balanced Accuracy (General): {balanced_accuracy_score(labels, predictions)}')

    # Separamos según la calidad de la etiqueta.

    # Confiable
    reliable_mask = df_test.label_quality == 'reliable'
    X_reliable = sentences[reliable_mask]
    y_reliable = labels[reliable_mask]

    probabilities = model.predict(X_reliable, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    print(f'Balanced Accuracy (Reliable): {balanced_accuracy_score(y_reliable, predictions)}')

    # No Confiable
    unreliable_mask = df_test.label_quality == 'unreliable'
    X_unreliable = sentences[unreliable_mask]
    y_unreliable = labels[unreliable_mask]

    probabilities = model.predict(X_unreliable, verbose=0)
    predictions = np.argmax(probabilities, axis=-1)
    print(f'Balanced Accuracy (Unreliable): {balanced_accuracy_score(y_unreliable, predictions)}')

