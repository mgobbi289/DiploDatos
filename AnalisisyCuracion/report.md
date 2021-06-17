## Criterios de exclusión
  1. Se eliminan variables que poseen un gran porcentaje de nulos, como BuildingArea y YearBuilt.
  2. Eliminamos variables que no poseen informacion util como Address, Bedroom2
  3. Eliminamos registros que estan por encima del 97.5 y debajo del 2.5 con respecto a la variable Price.

## Características seleccionadas
  ### Características categóricas
  
- `Suburb` Barrio de residencia.  
  Lamentablemente no contamos con los recursos necesarios para correr KNN sobre nuestro conjunto de datos entero. Por lo tanto se tomó la decisión de eliminarla porque posee un gran cantidad de categorias
- `Type` Tipo de la propiedad.
  En orden de cantidad de ocurrencias, *house*, *unit*, y *townhouse*.
- `Method` Forma de adquisión de la propiedad.
  En orden de cantidad de ocurrencias, *property sold*, *property sold prior*, *property passed in*, *vendor bid*, y *sold after auction*.
  La última categoría tiene muy pocos valores, pero no se puede agrupar con ningún otra ya que es la única.
- `Regionname` Región general de la propiedad.
- `CouncilArea` Departamento (o algo similar a lo que entendemos).  
- `Date` Fecha de venta de la propiedad.  
- `SellerG` Vendedor de la propiedad.  
  Lamentablemente no contamos con los recursos necesarios para correr KNN sobre nuestro conjunto de datos entero. Por lo tanto se tomó la decisión de eliminarla porque posee un gran cantidad de categorias
- `bnb_city_mode` Ciudad de la propiedad  
   Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge, y realizar la operacion Moda
- `bnb_suburb_mode` Barrio de la residencia  
   Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge, y realizar la operacion Moda   
   Lamentablemente no contamos con los recursos necesarios para correr KNN sobre nuestro conjunto de datos entero. Por lo tanto se tomó la decisión de eliminarla porque posee un gran cantidad de categorias
   
Todas las características categóricas fueron codificadas con un método OneHotEncoding.

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
- `Lattitude`: Latitud de ubicación.
- `Longtitude`: Longitud de ubicación.
- `Bathroom` Cantidad de baños.
- `Car` Cantidad de cocheras.
  La variable tiene 0s, pero serán considerados normales ya que es razonable tener una propiedad sin cochera.
  La variable tiene NaNs, los cuales serán imputados por el valor constante 0.
  Si fuese una variable categórica, se podrían agrupar todas las propiedades con 5 o más cocheras.
- `Landsize` Tamaño del Terreno.  
  La variable tiene 0s, que asumiremos normales ya que no todas las propiedades pueden tener patios.
  La *media* es aproximadamente 550.
- `zipcode` Codigo postal.  
  Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge
- `bnb_price_mean` Precio promedio de la propiedad  
  Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge
- `bnb_latitude_mean` Latitud promedio de ubicación  
  Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge  
- `bnb_longitude_mean` Longitud promedio de ubicacion  
  Proviene del conjunto de datos de AirBnb luego de aplicar el metodo Merge

Todas las características numéricas fueron codificadas con un método OneHotEncoding.

### Transformaciones
  1. La columna `Bathroom` fue imputada para los valores que eran 0 por 1, por logica de que una propiedad siempre posee al menos un baño.
  2. La columna `Car` fue imputada utilizando el metodo FillNa, dandole valores 0.
  3. Agrupamos la variable `RegionName` dado que la region Victoria, se encontraba desgregada (Eastern, Northern, Western).
  4. Agrupamos la variable `Date` en forma trimestral.
  5. Agrupamos la variable `SellerG` donde los que que figuraban con 1 propiedad se generalizaron en Others
  3. Las columnas `YearBuilt` y `BuildingArea` fueron imputadas utilizando el 
     algoritmo IterativeImputer con un estimador KNeighborsRegressor

### Datos aumentados
  1. Se agregan las 5 primeras columnas obtenidas a través del
     método de PCA, aplicado sobre el conjunto de datos
     totalmente procesado.