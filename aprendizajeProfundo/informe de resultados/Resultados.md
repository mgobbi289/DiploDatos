# **Práctico Deep Learning**
## **intro**
En este práctico trabajaremos en el problema de clasificación de texto del **MeLi Challenge 2019.**

El datasets tiene información acerca de títulos de publicaciones, categoría de los mismos, información de idioma y confiabilidad de la anotación.
Cuenta con anotaciones de títulos para 632 categorías distintas.

El conjunto de entrenamiento tiene más de 4.800.000 de publicaciones.

El dataset también cuenta con una partición de test que está compuesta de 63.680 ejemplos con las mismas categorías.

El dataset incluye también datos en idioma portugues.

**Integrantes**
* Cecilia Garcia
* Bernaschini Laura
* Cámara Florencia
* Ferrer Rodrigo - DNI 36356441
* Gobbi Matias

## **Procedimiento general**
- Se decidio trabajar con el dataset en Español, si bien se puede establecer el idioma como otro parametro.
- Se parte del dataset preprocesado con los titulos de las publicaciones vectorizadas con los indices de las palabras significativas y con la etiqueta tambien procesada.
- Para manejar el dataset se utilizó el script dataset.py donde se crea una clase con base a **"IterableDataset"** con la posibilidad de otorgar el tamaño del buffer para trabajar.
- En el script utils.py se establece la clase **"PadSequences"** para otorgarle la misma longitud a todos los vectores dentro del batch.
- Se establecieron 3 clases distintas de modelos **(MLP, CNN y RNN)**, con la posibilidad de instaciar cada uno de ellos con los distintos hiperparametros y con algunos hiperparametros predefinidos como baseline.
- Se probaron distintas comobinaciones de hiperparametros para cada clase de red.
- Se utilizó MLFLOW para guardar los **Logs** de entrenamiento y validación.
- Se comprobó para cada uno la performance en el conjunto de test.
- Por ultimo se implementaron modelos para busqueda automatica de hiperparametros.

## **Reporte de experimentos**

- baseline_MLP
- baseline_CNN
- baseline_RNN

CNN 1:     
    --language spanish \
    --hidden-layers 512 256 128 \
    --dropout 0.3 \
    --learning-rate 1e-4 \
    --random-buffer-size 2048 \
    --epochs 15 \
    --batch-size 300 \
    --filters-length 2 3 4\
    --filters-count 200 \

<img src='imagenes/+_mlflow1.png' alt='MeLi Challenge spanish' style='float: center; margin-right: 10px;' width='80%'/>
test_baccuracy=0.722
<img src='imagenes/+_mlflow2-loss.png' alt='MeLi Challenge spanish' style='float: center; margin-right: 10px;' width='80%'/>