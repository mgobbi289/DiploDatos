# AutoML: Práctico nº1
## Hyperopt: Distributed Hyperparameter Optimization

Librería de Python de código abierto creada por **James Bergstra en 2011**.

Se creó para optimizar **pipelines** de machine learning, incluyendo el preproceso de los datos, la selección del modelo y los hiperparametros de este.

Permite automatizar la búsqueda de los **hiperparámetros óptimos** de un modelo de aprendizaje automático. Basada en una **Optimización Bayesiana** y soportada por la metodología **SMBO** (Sequential Model-Based Global Optimization) 
adaptada para trabajar con diferentes algoritmos tales como: Árbol de Estimadores Parzen (**TPE**), Árbol de adaptación de Estimadores Parzen (**ATPE**) y Procesos Gaussianos (**GP**).

HyperOpt toma la Optimización Bayesiana como premisa al realizar algunas variaciones en el proceso de muestreo, la definición y reducción del espacio de búsqueda y los algoritmos para maximizar el modelo de probabilidad.

Requiere 4 componentes esenciales para la optimización de los hiperparámetros: el **espacio de búsqueda** , la **función de pérdida** , el **algoritmo de optimización** y una **base de datos** para almacenar el historial.

Permite escalar el procedimiento de optimización en **múltiples núcleos y múltiples máquinas** (Apache Spark y MongoDB)
![Texto alternativo](https://github.com/mgobbi289/DiploDatos/blob/main/AutoML/Imagenes/1_ztfyT1QatezmRHx4Zjeq5g.jpeg)
Se creó una extensión de HyperOpt llamada **HyperOpt-Sklearn** que permite aplicar el procedimiento HyperOpt a la preparación de datos y los modelos de aprendizaje automático proporcionados por Scikit-Learn
![Texto alternativo](https://github.com/mgobbi289/DiploDatos/blob/main/AutoML/Imagenes/1_b1zNb0WFu5j-B01NROGDCQ.jpeg)

~~~
# define an objective function
def objective(args):
    case, val = args
    if case == 'case 1':
        return val
    else:
        return val ** 2

# define a search space
from hyperopt import hp
space = hp.choice('a',
    [
        ('case 1', 1 + hp.lognormal('c1', 0, 1)),
        ('case 2', hp.uniform('c2', -10, 10))
    ])

# minimize the objective over the space
from hyperopt import fmin, tpe
best = fmin(objective, space, algo=tpe.suggest, max_evals=100)

print best
# -> {'a': 1, 'c2': 0.01420615366247227}
print hyperopt.space_eval(space, best)
# -> ('case 2', 0.01420615366247227}
~~~


Presentación:https://docs.google.com/presentation/d/1EG_cKrkJaAEv6h6SHR7VchyyDrqYoSVS_vSlev0Yio8/edit#slide=id.gc6f90357f_0_47

Referencias:
http://hyperopt.github.io/hyperopt/
https://github.com/hyperopt/hyperopt
https://ichi.pro/es/hyperopt-ajuste-de-hiperparametros-basado-en-optimizacion-bayesiana-140338828128041
https://ichi.pro/es/introduccion-a-la-optimizacion-automatica-de-hiperparametros-con-hyperopt-247088534065241
https://machinelearningmastery.com/hyperopt-for-automated-machine-learning-with-scikit-learn/
https://proceedings.mlr.press/v28/bergstra13.html
