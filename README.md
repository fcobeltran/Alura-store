# Análisis de Tiendas Alura Store

Este proyecto realiza un análisis completo de los datos de ventas de cuatro tiendas pertenecientes a Alura Store. El objetivo es ayudar al Sr. Juan a decidir qué tienda debe vender para invertir en un nuevo negocio, basándose en un análisis comparativo del desempeño de cada una.

## Métricas analizadas

El análisis se centra en cinco aspectos clave:

1. **Facturación total** de cada tienda
2. **Categorías más populares** en cada tienda
3. **Promedio de evaluación** de clientes por tienda
4. **Productos más y menos vendidos** en cada tienda
5. **Costo promedio de envío** por tienda

## Requisitos

Para ejecutar este análisis, necesitas tener instaladas las siguientes bibliotecas de Python:

```
pandas
matplotlib
seaborn
numpy
```

Puedes instalarlas con el siguiente comando:

```
pip install pandas matplotlib seaborn numpy
```

## Ejecución del análisis

Para ejecutar el análisis completo, simplemente ejecuta el script `analisis_alurastore.py`:

```
python analisis_alurastore.py
```

## Resultados

El script generará visualizaciones en formato PNG para cada una de las métricas analizadas:

- `1_facturacion_total.png`: Gráfico de barras con la facturación total por tienda
- `2_categorias_populares_global.png`: Comparativa de las categorías más populares entre tiendas
- `2_categorias_populares_detalle.png`: Desglose de categorías populares por tienda
- `3_evaluacion_clientes.png`: Promedio de evaluación de clientes por tienda
- `4_productos_vendidos.png`: Productos más y menos vendidos por tienda
- `5_costo_envio.png`: Costo promedio de envío por tienda
- `6_puntuacion_total.png`: Puntuación total que integra todas las métricas

Además, el script mostrará en la consola un resumen detallado de cada métrica y una recomendación final sobre qué tienda debería vender el Sr. Juan.

## Datos utilizados

Los datos se obtienen directamente de archivos CSV alojados en GitHub:

- Tienda 1: `tienda_1.csv`
- Tienda 2: `tienda_2.csv`
- Tienda 3: `tienda_3.csv`
- Tienda 4: `tienda_4.csv`

Estos archivos son parte del desafío de análisis de datos de Alura LATAM.

## Metodología

El análisis normaliza todas las métricas para permitir una comparación justa entre las tiendas. La puntuación total se calcula ponderando cada métrica según su importancia relativa:

- Facturación total: 35%
- Evaluación de clientes: 25%
- Volumen de ventas: 25%
- Costo de envío: 15%

La tienda con la puntuación total más baja se recomienda como candidata para vender. 