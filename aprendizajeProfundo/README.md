# Aprendizaje Profundo: Práctico

En este práctico trabajaremos en el problema de clasificación de texto del **MeLi Challenge 2019**.

<img src='images/+_Practico-1.png' alt='MeLi Challenge Spanish' style='float: center; margin-right: 10px;' width='80%'/>

El dataset tiene información acerca de títulos de publicaciones, categoría de los mismos, información de idioma, y confiabilidad de la anotación.
Cuenta con anotaciones de títulos para **632** categorías distintas.

<img src='images/+_Practico-2.png' alt='Categorias' style='float: center; margin-right: 10px;' width='50%'/>

El dataset también cuenta con una partición de evaluación que está compuesta de **63680** ejemplos con las mismas categorías
(aunque no necesariamente la misma distribución).

También hay datos en idioma portugués, aunque para el práctico de esta materia basta con usar uno solo de los idiomas.

<img src='images/+_Practico-3.png' alt='MeLi Challenge Portuguese' style='float: center; margin-right: 10px;' width='80%'/>

## Ejercicio:

Implementar una red neuronal que asigne una categoría dado un título.
Para este práctico se puede usar cualquier tipo de red neuronal.
Los que hagan solo la primera mitad de la materia, implementarán un **MLP**.
Quienes cursan la materia completa, deberían implementar algo más complejo, usando **CNNs**, **RNNs**, o **Transformers**.

<img src='images/+_Practico-4.png' alt='NN Generic Architecture' style='float: center; margin-right: 10px;' width='40%'/>

Algunas consideraciones a tener en cuenta para estructurar el trabajo:

  1. Hacer un preprocesamiento de los datos (¿Cómo vamos a representar los datos de entrada? ¿Y las categorías?).
  2. Tener un manejador del dataset (alguna clase o función que nos divida los datos en *batches*).
  3. Crear una clase para el modelo que se pueda instanciar con diferentes hiperparámetros.
  4. Hacer *logs* de entrenamiento (reportar tiempo transcurrido, iteraciones, loss, accuracy, etc.). Usar **MLFlow**.
  5. Hacer un gráfico de la función de *loss* a lo largo de las *epochs*. **MLFlow** también puede generar la gráfica.
  6. Reportar performance en el conjunto de test con el mejor modelo entrenado. La métrica para reportar será la *balanced accuracy* ([Macro-recall](https://peltarion.com/knowledge-center/documentation/evaluation-view/classification-loss-metrics/macro-recall)).

## Ejercicios opcionales:

  1. Se puede tomar un subconjunto del entrenamiento para armar un conjunto de evaluación para probar configuraciones, arquitecturas, o hiperparámetros (cuidado con la distribución de este conjunto). **No usar el conjunto de test para ajustar hiperparámetros**.
  2. Se pueden usar todos los datos (español y portugués) para hacer un modelo multilingual para la tarea. ¿Qué cosas tendrían que tener en cuenta para ello?

Adicionalmente, se pide un reporte de los experimentos y los procesos que se llevaron a cabo (en el **README.md** de su repositorio correspondiente).
No se evaluará la performance de los modelos, sino el proceso de tomar el problema e implementar una solución con aprendizaje profundo.