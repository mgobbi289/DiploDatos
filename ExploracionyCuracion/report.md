## Criterios de exclusión
  1. Eliminamos variables que poseen un gran porcentaje de nulos, como `BuildingArea` y `YearBuilt`.
  2. Eliminamos variables que no poseen información útil, como `Address` y `Bedroom2`.
  3. Ignoramos variables que poseen un gran porcentaje de nulos, como `weekly_price` y `monthly_price`.
  4. Ignoramos variables con demasiadas categorías, como `description` y `neighborhood_overview`.
  5. Ignoramos variables que no poseen información útil, como `street` y `state`.
  6. Ignoramos variables que no fue posible imputar, como `neighborhood`.
  7. Ignoramos variables para reducir la dimensionalidad del conjunto de datos procesado, como `SellerG`, `Suburb`, y `bnb_suburb_mode`.

## Características seleccionadas

### Características categóricas
  - `Type` Tipo de la propiedad.
  En orden de cantidad de ocurrencias, *house*, *unit*, y *townhouse*.
  - `Method` Forma de adquisión de la propiedad.
  En orden de cantidad de ocurrencias, *property sold*, *property sold prior*, *property passed in*, *vendor bid*, y *sold after auction*.
  - `Regionname` Región general de la propiedad.
  - `CouncilArea` Departamento (o algo similar a lo que entendemos).
  - `Date` Fecha de venta de la propiedad.
  - `bnb_city_mode` Ciudad de la propiedad.
  Proviene del conjunto de datos de AirBnB luego de aplicar el método `merge`, y realizar la operación `mode`.

Todas las variables categóricas fueron codificadas con un método `OneHotEncoding`.

### Características numéricas
  - `Price` Precio de la propiedad.
  La *media* es aproximadamente 1075000.
  - `Distance` Distancia al centro.
  Hay valores con cero, pero serán considerados normales ya que es razonable vivir en el centro.
  La *media* es aproximadamente 10.
  - `Propertycount` Cantidad de propiedades en el suburbio.
  La *media* es aproximadamente 7450.
  - `Postcode` Código postal.
  - `Rooms` Cantidad de habitaciones.
  - `Lattitude` Latitud de ubicación.
  - `Longtitude` Longitud de ubicación.
  - `BuildingArea` Tamaño de la edificación.
  La *media* es aproximadamente 150.
  - `YearBuilt` Año de construcción.
  - `Bathroom` Cantidad de baños.
  - `Car` Cantidad de cocheras.
  La variable tiene 0s, pero serán considerados normales ya que es razonable tener una propiedad sin cochera.
  - `Landsize` Tamaño del Terreno.
  La variable tiene 0s, que asumiremos normales ya que no todas las propiedades pueden tener patios.
  La *media* es aproximadamente 550.
  - `zipcode` Código postal.
  Proviene del conjunto de datos de AirBnB luego de aplicar el método `merge`.
  - `bnb_price_mean` Precio promedio de la propiedad.
  Proviene del conjunto de datos de AirBnB luego de aplicar el método `merge`, y realizar la operación `mean`.
  - `bnb_latitude_mean` Latitud promedio de ubicación.
  Proviene del conjunto de datos de AirBnB luego de aplicar el método `merge`, y realizar la operación `mean`.
  - `bnb_longitude_mean` Longitud promedio de ubicación.
  Proviene del conjunto de datos de AirBnB luego de aplicar el método `merge`, y realizar la operación `mean`.

Todas las variables numéricas fueron codificadas con un método `OneHotEncoding`.

### Transformaciones
  1. La columna `Bathroom` fue imputada para los valores que eran $0$ por $1$, por lógica que una propiedad siempre posee al menos un baño.
  2. La columna `Car` fue imputada utilizando el método `fillna`, asignando valores 0.
  3. Agrupamos la variable `Regionname` dado que la región *Victoria*, se encontraba desgregada (*Eastern*, *Northern*, *Western*).
  4. Agrupamos la variable `Date` en forma trimestral.
  5. Agrupamos la variable `SellerG` donde los que figuraban con una única propiedad se generalizaron en *Others*.
  6. Limpiamos la variable `price`, eliminando los *outliers* que se encuentran a 2.5 desviaciones estándar de la media.
  7. Unimos nuestro conjunto de datos con el conjunto de datos de [AirBnB](https://www.kaggle.com/tylerx/melbourne-airbnb-open-data?select=cleansed_listings_dec18.csv), empleando las variables `Postcode` y `zipcode` respectivamente.
  8. Al unir los conjuntos de datos, agregamos la variable `price` por el promedio.
  9. Al unir los conjuntos de datos, agregamos la variable `latitude` por el promedio.
  10. Al unir los conjuntos de datos, agregamos la variable `longitude` por el promedio.
  11. Al unir los conjuntos de datos, agregamos la variable `city` por la moda.
  12. Imputamos los valores faltantes de `CouncilArea`, según la información almacenada en la variable `Suburb`.
  13. Imputamos los valores faltantes de `CouncilArea`, por la moda.
  14. Imputamos los valores faltantes de `bnb_city_mode`, por la moda.
  15. Imputamos los valores faltantes de `zipcode`, por la media.
  16. Imputamos los valores faltantes de `bnb_price_mean`, por la media.
  17. Imputamos los valores faltantes de `bnb_latitude_mean`, por la media.
  18. Imputamos los valores faltantes de `bnb_longitude_mean`, por la media.
  19. Aplicamos `OneHotEncoder` a todo nuestro conjunto de datos.
  20. Estandarizamos por `StandardScaler` a todo nuestro conjunto de datos.
  21. Imputamos las variables `BuildingArea` y `YearBuilt`, con el algoritmo `IterativeImputer` y el estimador `KNeighborsRegressor`, utilizando todo nuestro conjunto de datos.

### Datos aumentados
  1. Se agregan los $20$ primeros componentes principales, obtenidos al aplicar `PCA` sobre el conjunto de datos totalmente procesado.
