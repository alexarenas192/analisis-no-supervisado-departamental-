+Proyecto  de Tienda Departamental

+ Descripción del Proyecto

Este proyecto se enfoca en el análisis no supervisado de datos de una tienda departamental para segmentar a sus clientes. A pesar de manejar grandes volúmenes de información sobre ventas y clientes, la tienda no estaba aprovechando estos datos para entender el comportamiento del consumidor y tomar decisiones estratégicas.

El objetivo principal es descubrir patrones ocultos en el historial de compras de los clientes monto total gastado, frecuencia de compra, cantidad de productos adquiridos para agruparlos en distintos perfiles. Esta segmentación permitirá a la tienda diseñar estrategias  más personalizadas y mejorar la fidelización y aumentar la rentabilidad.

* Algoritmos Aplicados

Para este análisis se utilizaron los siguientes algoritmos:
---K-Means Clustering:** Empleado para agrupar a los clientes en clústeres basados en sus características de compra.
--PCA Análisis de Componentes Principales:** Utilizado para reducir la dimensionalidad de los datos y permitir la visualización gráfica de los clústeres resultantes.

* Flujo de Trabajo

El proceso de desarrollo incluyó los siguientes pasos clave:
1.  Carga del dataset `ventas_tienda_limpio.csv`.
2.  Selección de variables relevantes para el clustering (monto, cantidad, frecuencia).
3.  Normalización de datos utilizando `StandardScaler`.
4.  Aplicación iterativa del algoritmo K-Means con diferentes valores de K.
5.  Evaluación del modelo usando métricas no supervisadas (Silhouette Score, Calinski-Harabasz Index, Davies-Bouldin Index).
6.  Optimización y selección del número óptimo de clústeres (K=4) basado en las métricas y el Método del Codo.
7.  Visualización de los clústeres resultantes mediante PCA.

* Resultados Clave

El análisis resultó en la identificación de 4 perfiles de clientes distintivos:
-   **Clúster 0: Clientes de Bajo Consumo**
-   **Clúster 1: Clientes Frecuentes Medios**
-   **Clúster 2: Clientes Ocasionales de Valor**
-   **Clúster 3: Clientes de Alto Valor (VIP)**

Estos perfiles proporcionan una base sólida para estrategias comerciales personalizadas.

*Visualizaciones:

*Gráfico del Método del Codo:** Muestra la suma de cuadrados intra-clúster (WSS) para diferentes K, ayudando a determinar el óptimo.

*Cómo Ejecutar el Proyecto Concepto

Para replicar el análisis, se necesitarán las librerías listadas en `requirements.txt`. El código principal de segmentación se encuentra en `customer_segmentation.py`.

*Documentación

El informe completo del proyecto, incluyendo detalles metodológicos, resultados y conclusiones, se encuentra en `Informe_Analisis_No_Supervisado.pdf`.