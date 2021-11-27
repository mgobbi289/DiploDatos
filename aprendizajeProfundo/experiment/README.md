# Aprendizaje Profundo: Entrega

En este trabajo nos enfrentamos al problema de clasificación de texto del **MeLi Challenge 2019**.

El conjunto de datos tiene información acerca de títulos de publicaciones, la correspondiente categoría a predecir para cada una, información del idioma (podría ser en español o portugués), y la confiabilidad de la etiqueta.
En total, cuenta con anotaciones de títulos para 632 categorías distintas.

## **Procedimiento General**

#### Preprocesamiento
- En la notebook `preprocess_MeLi.ipynb` se aplica un preprocesamiento general a todo el conjunto de datos.
    - Se convierten los títulos de las publicaciones en vectores de índices (correspondientes a la ubicación de cada palabra en el diccionario de vocabulario), y se transforman las categorías a valores numéricos.
- Para el manejo del dataset se utilizó el script `dataset.py` donde se hereda de la clase `IterableDataset`, agregando un hiperparámetro para determinar el tamaño del buffer de pseudo-aleatorización.
- Para asegurar que todas las entradas al modelo tengan las mismas dimensiones, en el script `utils.py` se define la clase `PadSequences` para asignar la misma longitud a todos los vectores de entrada.

#### Modelos
- Se declararon tres arquitecturas generales diferentes **MLP**, **CNN**, y **RNN**, donde cada una tiene la posibilidad de ser instanciada por un conjunto de hiperparámetros.
    - Se proveen valores por defectos para cada uno de los hiperparámetros, los cuales determinan nuestros *baselines*.
- Se utilizó **MLFlow** para almacenar *logs* y resultados para cada uno de los modelos definidos, para los conjuntos de entrenamiento, validación, y evaluación.
    - Se probaron distintas combinaciones de hiperparámetros para cada arquitectura.
    - La métrica de medición era la *balanced accuracy* en el conjunto de evaluación.

#### Decisiones
- Se decidió limitar el trabajo al dataset en español, de todas maneras se puede establecer el idioma del modelo como otro hiperparámetro.
- Partiendo de los *baselines*, se intentó encontrar modelos superadores.
    - Inicialmente aplicando una búsqueda de hiperparámetros automática.
    - Finalmente aplicando una búsqueda de hiperparámetros manual.

#### Dificultades

- El conjunto de datos es demasiado grande, en consecuencia los modelos empleados también lo serán, por lo que no era posible ejecutar los experimentos en nuestras computadoras personales. 
- Al no contar con los recursos necesarios, se decidió utilizar (y compartir) el servidor provisto por la universidad para encolar tareas.
- Debido que los entrenamientos, las validaciones, y las evaluaciones eran extensas, la disponibilidad de tiempo fue muy escasa; por lo que se decidió limitar el alcance de nuestras experimentaciones.

A continuación reportaremos los resultados de nuestros experimentos, junto con algunas reflexiones y observaciones sobre la investigación realizada.

# Baselines

Existe un conjunto de hiperparámetros que se mantuvo constante durante el desarrollo de todo el trabajo.
- Los conjuntos de datos `train_data`, `validation_data`, `test_data`.
- La preparación de los *embeddings* `language`, `embeddings`.
- Los resultados del preprocesamiento `token_to_index`.
- La pseudo-aleatorización de los datos `random_buffer_size`.

#### Hiperparámetros

| Model                | `hidden_layers`      | `dropout`            | `learning_rate`      | `weight_decay`       | `epochs`             | `batch_size`         | `freeze_embeddings`  | `filters_count`      | `filters_length`     | `lstm_layers`        | `lstm_features`      |
| -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| MLP                  | (256, 128)           | 0.3                  | 0.001                | 1e-05                | 3                    | 128                  | True                 | ---                  | ---                  | ---                  | ---                  |
| CNN                  | (128)                | 0.0                  | 0.001                | 1e-05                | 3                    | 128                  | True                 | 100                  | (2, 3, 4)            | ---                  | ---                  |
| RNN                  | (256, 128)           | 0.3                  | 0.001                | 1e-05                | 3                    | 128                  | True                 | ---                  | ---                  | 3                    | 128                  |

#### Métricas

| Model                     | Test Balanced Acc.        | Validation Balanced Acc.  | Test Loss                 | Validation Loss           | Train Loss                | 
| ------------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| MLP                       | **0.447**                 | 0.414                     | 2.768                     | 2.810                     | 2.834                     |
| CNN                       | **0.725**                 | 0.675                     | 1.220                     | 1.433                     | 1.437                     |
| RNN                       | **0.870**                 | 0.795                     | 0.574                     | 0.869                     | 0.968                     |

#### Observaciones

- Resulta evidente que nuestro mejor modelo es **RNN** (mientras que el peor es **MLP**).
- A pesar de solo ser *baselines*, se obtuvieron resultados bastante prometedores.
- Notar que varios de los hiperparámetros de los modelos coinciden en sus valores.

# Búsqueda de Hiperparámetros

Definiendo un espacio totalmente arbitrario de búsqueda, se intentó automatizar el proceso para encontrar mejores hiperparámetros.
- Se utilizó la librería `hyperopt`, y se realizó un máximo de **10** intentos por cada arquitectura.
En caso de ser necesario, se puede revisar el script correspondiente a cada modelo para verificar el espacio considerado en los experimentos.

#### Hiperparámetros

| Model                | `hidden_layers`      | `dropout`            | `learning_rate`      | `weight_decay`       | `epochs`             | `batch_size`         | `freeze_embeddings`  | `filters_count`      | `filters_length`     | `lstm_layers`        | `lstm_features`      |
| -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| MLP                  | (256, 128)           | 0.2                  | 6e-04                | 1e-05                | 10                   | 514                  | False                | ---                  | ---                  | ---                  | ---                  |
| CNN                  | (256, 128)           | 0.4                  | 5e-05                | 1e-05                | 1                    | 514                  | False                | 100                  | (2, 3, 4, 5)         | ---                  | ---                  |
| RNN                  | (256, 514, 256)      | 0.4                  | 8e-05                | 1e-05                | 10                   | 128                  | True                 | ---                  | ---                  | 3                    | 128                  |

#### Métricas

| Model                     | Test Balanced Acc.        | Validation Balanced Acc.  | Test Loss                 | Validation Loss           | Train Loss                | 
| ------------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- | ------------------------- |
| MLP                       | **0.932**                 | 0.875                     | 0.279                     | 0.521                     | 0.502                     |
| CNN                       | **0.563**                 | 0.522                     | 2.119                     | 1.893                     | 3.849                     |
| RNN                       | **0.675**                 | 0.600                     | 1.493                     | 1.720                     | 1.837                     |

#### Observaciones

- Notar que solo listamos el mejor modelo encontrado por cada búsqueda.
- Resulta evidente que la búsqueda no fue del todo adecuada, o al menos no brindó los resultados esperados.
    - Quizás el espacio de búsqueda era demasiado grande (muchos parámetros y muchas opciones).
    - Probablemente se realizaron muy poco intentos, y se necesitarían más para que el proceso de búsqueda comenzara a converger a mejores modelos.
    - Incluso es posible que las opciones especificadas no sean lo suficientemente buenas para encontrar un modelo convincente.
- De todas maneras, se encontró un modelo extremadamente satisfactorio de **MLP** (superando ampliamente su *baseline*).

# TODO: Redactar cada una de las siguientes secciones...

# Experimentos Manuales

Enumerar varios experimentos manuales (cambiando algunos hiperparámetros de forma totalmente arbitraria) que decidimos probar.

Listado de resultados y métricas.
- Balanced Accuracy en test y train.
- Loss en test y train.

(mencionar los hiperparámetros modificados en cada situación obviamente)

# Conclusión
